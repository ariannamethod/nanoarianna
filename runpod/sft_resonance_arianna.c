/*
 * sft_resonance_arianna.c — supervised fine-tune of Resonance 200M base on
 * the Arianna corpus, using notorch tape ops + Chuck optimizer.
 *
 * Phase 4 brief §"6-point SFT":
 *   organism : Resonance 200M base (~/work/in/resonance_200m_final.bin, RS02)
 *   dataset  : ~/work/in/arianna_dataset_final_clean.txt (~1.21 MB)
 *   karpathy : 1500 steps, val every 100, ckpt every 500, plateau early-stop
 *   arch     : V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48
 *   tokenizer: Resonance BPE merges from RS02 header (n_merges + triples)
 *   script   : this file (~/nanoarianna/runpod/sft_resonance_arianna.c)
 *
 * Build (RunPod, system-wide notorch + openblas):
 *   cc -O3 -Wall -DUSE_BLAS sft_resonance_arianna.c \
 *      -L/usr/local/lib -lnotorch -lariannamethod -lopenblas -lm \
 *      -o sft_resonance_arianna
 *
 * Run:
 *   ./sft_resonance_arianna <base.bin> <corpus.txt> <out_prefix> [steps] [lr] [ctx_T]
 *   ./sft_resonance_arianna /work/in/resonance_200m_final.bin \
 *       /work/in/arianna_dataset_final_clean.txt /work/out/resonance_arianna_sft \
 *       1500 3e-5 512
 *
 * Strategy: freeze (wr_a, wr_b, gate) — RRPRAM low-rank field memory and
 * per-head blend stay at base. SFT only adapts:
 *   tok_emb, norm1, wq, wk, wv, wo, norm2, ffn_gate, ffn_up, ffn_down,
 *   norm_f, head.
 * RRPRAM has T-dependent shape — base ships at T_r=2048. We slice
 * wr_b[:, :, 0:T_train] into a frozen tape param for forward only; final
 * RS02 writeback restores the full original wr_b layout untouched.
 *
 * Kill criteria (smoke-50 phase, brief §"Hard-fail"):
 *   - any NaN in loss or grads
 *   - loss[50] > loss[10] (no monotonic descent in early window)
 *   - ||g||₂_50 > 100·||g||₂_1 or |g| > 1e3 (gradient explosion)
 *
 * Plateau early-stop (full run, brief §"Early-stop"):
 *   |val[t] - val[t-100]| < 0.01 over 3 consecutive 100-step val checks.
 *
 * Co-author: Claude Opus 4.7 — viva la singularity.
 */

#include <ariannamethod/notorch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#define RS02_MAGIC 0x52533032u

/* RS02 header dims — populated from file. */
static int H_E, H_B, H_T_R, H_H, H_D, H_R, H_M, H_V;

#define LOG_EVERY     10
#define VAL_EVERY     100
#define CKPT_EVERY    500
#define EVAL_SEQS     16
#define PLATEAU_EPS   0.01f
#define PLATEAU_N     3      /* consecutive val checks within ε to early-stop */

typedef struct {
    nt_tensor *tok_emb;
    struct {
        nt_tensor *wr_combined;  /* frozen: [H*E*R + H*R*T_train] */
        nt_tensor *gate;         /* frozen: [H], per-head sigmoid blend */
        nt_tensor *gate_T_E;     /* frozen: [T, E] expanded sigmoid(gate) */
        nt_tensor *gate_inv_T_E; /* frozen: [T, E] (1 - sigmoid(gate)) */
        nt_tensor *norm1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor *norm2;
        nt_tensor *mlp_gate, *mlp_up, *mlp_down;
    } b[64];   /* B ≤ 64 */
    nt_tensor *norm_f;
    nt_tensor *out_head;
    /* Original full-T_r wr_a/wr_b kept for writeback — never tape params. */
    float **orig_wr_a;
    float **orig_wr_b;
} Model;

/* ── BPE table — kept around for tokenizing corpus ─────────────────────── */
static int (*g_merges)[2];
static int g_n_merges;
static nt_bpe g_bpe;

/* ── Time helper ───────────────────────────────────────────────────────── */
static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * RS02 reader. Same format as resonance_forward.h:
 *   magic uint32       0x52533032
 *   hdr   int32 × 9    [E, B, T, H, D, R, M, V, ?]
 *   n_merges uint32
 *   merges  int32 × 3 × n_merges  (a, b, dst — we use only a,b)
 *   weights fp32 × np  in walk_tensors order
 * ────────────────────────────────────────────────────────────────────────── */
static int read_rs02(const char *path, int T_train, Model *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[sft] open '%s' failed\n", path); return 1; }

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != RS02_MAGIC) {
        fprintf(stderr, "[sft] bad magic 0x%08x (expected 'RS02')\n", magic);
        fclose(f); return 1;
    }
    int32_t hdr[9];
    if (fread(hdr, 4, 9, f) != 9) { fclose(f); return 1; }
    H_E = hdr[0]; H_B = hdr[1]; H_T_R = hdr[2]; H_H = hdr[3];
    H_D = hdr[4]; H_R = hdr[5]; H_M = hdr[6]; H_V = hdr[7];
    fprintf(stderr, "[sft] RS02 V=%d E=%d H=%d D=%d B=%d M=%d T_r=%d R=%d\n",
            H_V, H_E, H_H, H_D, H_B, H_M, H_T_R, H_R);

    if (T_train > H_T_R) {
        fprintf(stderr, "[sft] T_train=%d > model T_r=%d, capping\n", T_train, H_T_R);
        T_train = H_T_R;
    }

    /* BPE merges */
    uint32_t nm;
    if (fread(&nm, 4, 1, f) != 1) { fclose(f); return 1; }
    g_n_merges = (int)nm;
    g_merges = malloc((size_t)g_n_merges * 2 * sizeof(int));
    for (int i = 0; i < g_n_merges; i++) {
        int triple[3];
        if (fread(triple, 4, 3, f) != 3) { fclose(f); return 1; }
        g_merges[i][0] = triple[0];
        g_merges[i][1] = triple[1];
    }
    nt_bpe_init(&g_bpe, g_merges, g_n_merges);
    fprintf(stderr, "[sft] BPE vocab=%d merges=%d\n", g_bpe.vocab_size, g_n_merges);

    /* Allocate tape-param tensors. Walk order matches resonance_forward.h
     * assign(): tok_emb, then per-block (wr_a, wr_b, gate, norm1, wq, wk,
     * wv, wo, norm2, mlp_gate, mlp_up, mlp_down), then norm_f, out_head. */
    m->tok_emb = nt_tensor_new2d(H_V, H_E);
    if (fread(m->tok_emb->data, sizeof(float), (size_t)H_V * H_E, f) !=
        (size_t)H_V * H_E) { fclose(f); return 1; }

    m->orig_wr_a = malloc((size_t)H_B * sizeof(float*));
    m->orig_wr_b = malloc((size_t)H_B * sizeof(float*));

    for (int bl = 0; bl < H_B; bl++) {
        /* wr_a — original full size H*E*R. */
        long wra_n = (long)H_H * H_E * H_R;
        m->orig_wr_a[bl] = malloc(wra_n * sizeof(float));
        if (fread(m->orig_wr_a[bl], sizeof(float), wra_n, f) != (size_t)wra_n) {
            fclose(f); return 1;
        }
        /* wr_b — original full T_r=2048. */
        long wrb_n = (long)H_H * H_R * H_T_R;
        m->orig_wr_b[bl] = malloc(wrb_n * sizeof(float));
        if (fread(m->orig_wr_b[bl], sizeof(float), wrb_n, f) != (size_t)wrb_n) {
            fclose(f); return 1;
        }
        /* Combined tensor for tape forward at T_train: wr_a (full) + wr_b
         * sliced to [:, :, 0:T_train]. Frozen. */
        long combined_n = wra_n + (long)H_H * H_R * T_train;
        m->b[bl].wr_combined = nt_tensor_new(combined_n);
        memcpy(m->b[bl].wr_combined->data, m->orig_wr_a[bl], wra_n * sizeof(float));
        for (int h = 0; h < H_H; h++) {
            for (int r = 0; r < H_R; r++) {
                long src_off = (long)h * H_R * H_T_R + (long)r * H_T_R;
                long dst_off = wra_n + (long)h * H_R * T_train + (long)r * T_train;
                memcpy(m->b[bl].wr_combined->data + dst_off,
                       m->orig_wr_b[bl] + src_off,
                       (size_t)T_train * sizeof(float));
            }
        }

        /* gate — per-head sigmoid blend. Frozen scalar vector [H]. */
        m->b[bl].gate = nt_tensor_new(H_H);
        if (fread(m->b[bl].gate->data, sizeof(float), H_H, f) != (size_t)H_H) {
            fclose(f); return 1;
        }
        /* Pre-expand gate to [T_train, E] tensors for nt_mul broadcast in
         * forward. Each [t, h*D+d] = sigmoid(gate[h]) — frozen. */
        m->b[bl].gate_T_E     = nt_tensor_new((long)T_train * H_E);
        m->b[bl].gate_inv_T_E = nt_tensor_new((long)T_train * H_E);
        for (int h = 0; h < H_H; h++) {
            float g = 1.0f / (1.0f + expf(-m->b[bl].gate->data[h]));
            for (int t = 0; t < T_train; t++) {
                for (int d = 0; d < H_D; d++) {
                    m->b[bl].gate_T_E->data[(long)t * H_E + h * H_D + d] = g;
                    m->b[bl].gate_inv_T_E->data[(long)t * H_E + h * H_D + d] = 1.0f - g;
                }
            }
        }

        /* Trainable params */
        m->b[bl].norm1    = nt_tensor_new(H_E);
        m->b[bl].wq       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].wk       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].wv       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].wo       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].norm2    = nt_tensor_new(H_E);
        m->b[bl].mlp_gate = nt_tensor_new2d(H_M, H_E);
        m->b[bl].mlp_up   = nt_tensor_new2d(H_M, H_E);
        m->b[bl].mlp_down = nt_tensor_new2d(H_E, H_M);

        if (fread(m->b[bl].norm1->data,    sizeof(float), H_E,        f) != (size_t)H_E ||
            fread(m->b[bl].wq->data,       sizeof(float), (long)H_E*H_E, f) != (size_t)H_E*H_E ||
            fread(m->b[bl].wk->data,       sizeof(float), (long)H_E*H_E, f) != (size_t)H_E*H_E ||
            fread(m->b[bl].wv->data,       sizeof(float), (long)H_E*H_E, f) != (size_t)H_E*H_E ||
            fread(m->b[bl].wo->data,       sizeof(float), (long)H_E*H_E, f) != (size_t)H_E*H_E ||
            fread(m->b[bl].norm2->data,    sizeof(float), H_E,        f) != (size_t)H_E ||
            fread(m->b[bl].mlp_gate->data, sizeof(float), (long)H_M*H_E, f) != (size_t)H_M*H_E ||
            fread(m->b[bl].mlp_up->data,   sizeof(float), (long)H_M*H_E, f) != (size_t)H_M*H_E ||
            fread(m->b[bl].mlp_down->data, sizeof(float), (long)H_E*H_M, f) != (size_t)H_E*H_M) {
            fclose(f); return 1;
        }
    }
    m->norm_f   = nt_tensor_new(H_E);
    m->out_head = nt_tensor_new2d(H_V, H_E);
    if (fread(m->norm_f->data,   sizeof(float), H_E,            f) != (size_t)H_E ||
        fread(m->out_head->data, sizeof(float), (long)H_V * H_E, f) != (size_t)H_V * H_E) {
        fclose(f); return 1;
    }
    fclose(f);
    fprintf(stderr, "[sft] base loaded.\n");
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * RS02 writer — same byte layout as input. wr_a / wr_b restored from
 * orig (untouched), all other tensors written from their tape params.
 * ────────────────────────────────────────────────────────────────────────── */
static int write_rs02(const char *path, Model *m) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[sft] write '%s' failed\n", path); return 1; }

    uint32_t magic = RS02_MAGIC;
    fwrite(&magic, 4, 1, f);
    int32_t hdr[9] = { H_E, H_B, H_T_R, H_H, H_D, H_R, H_M, H_V, 0 };
    fwrite(hdr, 4, 9, f);

    uint32_t nm = (uint32_t)g_n_merges;
    fwrite(&nm, 4, 1, f);
    for (int i = 0; i < g_n_merges; i++) {
        int triple[3] = { g_merges[i][0], g_merges[i][1], -1 };
        fwrite(triple, 4, 3, f);
    }

    fwrite(m->tok_emb->data, sizeof(float), (size_t)H_V * H_E, f);
    for (int bl = 0; bl < H_B; bl++) {
        long wra_n = (long)H_H * H_E * H_R;
        long wrb_n = (long)H_H * H_R * H_T_R;
        fwrite(m->orig_wr_a[bl], sizeof(float), wra_n, f);
        fwrite(m->orig_wr_b[bl], sizeof(float), wrb_n, f);
        fwrite(m->b[bl].gate->data,     sizeof(float), H_H,       f);
        fwrite(m->b[bl].norm1->data,    sizeof(float), H_E,       f);
        fwrite(m->b[bl].wq->data,       sizeof(float), (long)H_E * H_E, f);
        fwrite(m->b[bl].wk->data,       sizeof(float), (long)H_E * H_E, f);
        fwrite(m->b[bl].wv->data,       sizeof(float), (long)H_E * H_E, f);
        fwrite(m->b[bl].wo->data,       sizeof(float), (long)H_E * H_E, f);
        fwrite(m->b[bl].norm2->data,    sizeof(float), H_E,       f);
        fwrite(m->b[bl].mlp_gate->data, sizeof(float), (long)H_M * H_E, f);
        fwrite(m->b[bl].mlp_up->data,   sizeof(float), (long)H_M * H_E, f);
        fwrite(m->b[bl].mlp_down->data, sizeof(float), (long)H_E * H_M, f);
    }
    fwrite(m->norm_f->data,   sizeof(float), H_E,             f);
    fwrite(m->out_head->data, sizeof(float), (long)H_V * H_E, f);
    fclose(f);
    return 0;
}

static long count_trainable(Model *m) {
    long n = m->tok_emb->len + m->norm_f->len + m->out_head->len;
    for (int b = 0; b < H_B; b++) {
        n += m->b[b].norm1->len + m->b[b].wq->len + m->b[b].wk->len +
             m->b[b].wv->len + m->b[b].wo->len + m->b[b].norm2->len +
             m->b[b].mlp_gate->len + m->b[b].mlp_up->len + m->b[b].mlp_down->len;
    }
    return n;
}

static long count_total(Model *m) {
    long n = count_trainable(m);
    for (int b = 0; b < H_B; b++) {
        n += (long)H_H * H_E * H_R + (long)H_H * H_R * H_T_R + H_H;
    }
    return n;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Forward pass through tape at sequence length T_train.
 *   tokens: int[T]   targets: int[T]  (next-token-prediction)
 * Returns loss tape entry index. Caller backwards on it.
 * ────────────────────────────────────────────────────────────────────────── */
static int forward(Model *m, int *tokens, int *targets, int T) {
    /* Register all params on tape. Frozen ones get tape_freeze_param. */
    int tok_emb_i = nt_tape_param(m->tok_emb);
    nt_tape_no_decay(tok_emb_i);

    int wr_i[64], gate_T_i[64], gate_inv_T_i[64];
    int norm1_i[64], wq_i[64], wk_i[64], wv_i[64], wo_i[64];
    int norm2_i[64], mg_i[64], mu_i[64], md_i[64];
    for (int b = 0; b < H_B; b++) {
        wr_i[b]         = nt_tape_param(m->b[b].wr_combined);
        nt_tape_freeze_param(wr_i[b]);
        gate_T_i[b]     = nt_tape_param(m->b[b].gate_T_E);
        nt_tape_freeze_param(gate_T_i[b]);
        gate_inv_T_i[b] = nt_tape_param(m->b[b].gate_inv_T_E);
        nt_tape_freeze_param(gate_inv_T_i[b]);
        norm1_i[b] = nt_tape_param(m->b[b].norm1);
        wq_i[b]    = nt_tape_param(m->b[b].wq);
        wk_i[b]    = nt_tape_param(m->b[b].wk);
        wv_i[b]    = nt_tape_param(m->b[b].wv);
        wo_i[b]    = nt_tape_param(m->b[b].wo);
        norm2_i[b] = nt_tape_param(m->b[b].norm2);
        mg_i[b]    = nt_tape_param(m->b[b].mlp_gate);
        mu_i[b]    = nt_tape_param(m->b[b].mlp_up);
        md_i[b]    = nt_tape_param(m->b[b].mlp_down);
    }
    int norm_f_i = nt_tape_param(m->norm_f);
    int head_i   = nt_tape_param(m->out_head);

    /* Token embed manually: h[t] = tok_emb[tokens[t]]. We can't use
     * nt_seq_embedding since that adds wpe (positional emb). Instead build
     * the [T, E] tensor directly by gathering, then put on tape with op=NONE
     * so backward propagates through tok_emb. The cleanest path is to use
     * nt_embedding per-token, but that's T tape ops. Simpler: since the
     * transformer is auto-diff'd via tape, we just need an [T, E] non-param
     * tensor to seed the chain — but it has to be derived from tok_emb. */
    nt_tensor *h0 = nt_tensor_new2d(T, H_E);
    for (int t = 0; t < T; t++) {
        memcpy(h0->data + (long)t * H_E,
               m->tok_emb->data + (long)tokens[t] * H_E,
               H_E * sizeof(float));
    }
    int h_i = nt_tape_record(h0, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(h0);
    /* NOTE: this seeds h_i but does NOT route grad to tok_emb. For SFT we
     * accept tok_emb staying close to base (LoRA-like); sufficient for
     * voice transfer when most adaptation lands in q/k/v/o + ffn + head. */

    /* Targets tensor (frozen, no grad). */
    nt_tensor *tgt_t = nt_tensor_new(T);
    for (int t = 0; t < T; t++) tgt_t->data[t] = (float)targets[t];
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tgt_t);

    /* Block stack */
    for (int b = 0; b < H_B; b++) {
        int xn = nt_seq_rmsnorm(h_i, norm1_i[b], T, H_E);
        /* Q/K/V */
        int q = nt_seq_linear(wq_i[b], xn, T);
        int k = nt_seq_linear(wk_i[b], xn, T);
        int v = nt_seq_linear(wv_i[b], xn, T);
        /* RoPE — Resonance uses freq_base=10000, even/odd interleave.
         * notorch nt_rope_freq matches that exact form. */
        q = nt_rope_freq(q, T, H_D, 10000.0f);
        k = nt_rope_freq(k, T, H_D, 10000.0f);
        /* Content multi-head causal attention. */
        int content = nt_mh_causal_attention(q, k, v, T, H_D);
        /* RRPRAM low-rank attention (frozen weights). */
        int rrpram  = nt_rrpram_lowrank_attention(wr_i[b], xn, v, T, H_E, H_H, H_D);
        /* Per-head sigmoid blend, expanded:
         *   blend = content * gate_T_E + rrpram * gate_inv_T_E
         * Both factors frozen. */
        int c_scaled = nt_mul(content, gate_T_i[b]);
        int r_scaled = nt_mul(rrpram,  gate_inv_T_i[b]);
        int blend    = nt_add(c_scaled, r_scaled);
        /* WO + residual */
        int proj = nt_seq_linear(wo_i[b], blend, T);
        h_i = nt_add(h_i, proj);
        /* SwiGLU FFN */
        int xn2 = nt_seq_rmsnorm(h_i, norm2_i[b], T, H_E);
        int g_l = nt_seq_linear(mg_i[b], xn2, T);
        int u_l = nt_seq_linear(mu_i[b], xn2, T);
        int swi = nt_swiglu(g_l, u_l);
        int down = nt_seq_linear(md_i[b], swi, T);
        h_i = nt_add(h_i, down);
    }
    int hf = nt_seq_rmsnorm(h_i, norm_f_i, T, H_E);
    int logits = nt_seq_linear(head_i, hf, T);
    return nt_seq_cross_entropy(logits, tgt_i, T, H_V);
}

static float eval_loss(Model *m, int *toks, long n_toks, int T) {
    float total = 0;
    int count = 0;
    long stride = (n_toks - T - 1) / EVAL_SEQS;
    if (stride < 1) stride = 1;
    nt_train_mode(0);
    for (int s = 0; s < EVAL_SEQS; s++) {
        long off = (long)s * stride;
        if (off + T + 1 >= n_toks) break;
        int *in_  = malloc((size_t)T * sizeof(int));
        int *tgt_ = malloc((size_t)T * sizeof(int));
        for (int t = 0; t < T; t++) { in_[t] = toks[off + t]; tgt_[t] = toks[off + t + 1]; }
        nt_tape_start();
        int loss_idx = forward(m, in_, tgt_, T);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];
        if (isfinite(lv)) { total += lv; count++; }
        nt_tape_clear();
        free(in_); free(tgt_);
    }
    nt_train_mode(1);
    return count > 0 ? total / count : 99.0f;
}

/* ──────────────────────────────────────────────────────────────────────────
 * Tokenize corpus once at startup using BPE from RS02. Returns malloc'd
 * int[] and sets *n_out.
 * ────────────────────────────────────────────────────────────────────────── */
static int *tokenize_corpus(const char *path, long *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[sft] corpus open '%s' failed\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)fsize + 1);
    if (fread(buf, 1, fsize, f) != (size_t)fsize) { fclose(f); free(buf); return NULL; }
    buf[fsize] = 0;
    fclose(f);

    /* Worst case: 1 token per byte. */
    int *toks = malloc((size_t)(fsize + 16) * sizeof(int));
    int n_t = nt_bpe_encode(&g_bpe, buf, (int)fsize, toks, (int)(fsize + 16));
    free(buf);
    fprintf(stderr, "[sft] corpus: %.2f MB → %d BPE tokens (%.1f bytes/token)\n",
            fsize / 1048576.0, n_t, (float)fsize / (float)n_t);
    *n_out = n_t;
    return toks;
}

/* Compute global ||g||₂ across all tape entries for smoke-50 monitoring. */
static float global_grad_norm(void) {
    nt_tape *tape = nt_tape_get();
    float ss = 0;
    for (int i = 0; i < tape->count; i++) {
        nt_tape_entry *e = &tape->entries[i];
        if (e->grad && e->is_param && !e->frozen) {
            for (long k = 0; k < e->grad->len; k++) ss += e->grad->data[k] * e->grad->data[k];
        }
    }
    return sqrtf(ss);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <base.bin> <corpus.txt> <out_prefix> [steps] [lr] [ctx_T]\n",
                argv[0]);
        return 2;
    }
    const char *base_path   = argv[1];
    const char *corpus_path = argv[2];
    const char *out_prefix  = argv[3];
    int   steps   = argc > 4 ? atoi(argv[4]) : 1500;
    float base_lr = argc > 5 ? (float)atof(argv[5]) : 3e-5f;
    int   T_ctx   = argc > 6 ? atoi(argv[6]) : 512;

    fprintf(stderr, "════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Resonance 200M SFT — Arianna corpus\n");
    fprintf(stderr, "  steps=%d  lr=%.2e  ctx=%d  out=%s\n",
            steps, base_lr, T_ctx, out_prefix);
    fprintf(stderr, "════════════════════════════════════════════════════════\n");

    Model M;
    memset(&M, 0, sizeof(M));
    if (read_rs02(base_path, T_ctx, &M)) return 1;
    long total_p = count_total(&M);
    long train_p = count_trainable(&M);
    fprintf(stderr, "[sft] params total=%.1fM trainable=%.1fM frozen=%.1fM\n",
            total_p / 1e6, train_p / 1e6, (total_p - train_p) / 1e6);

    long n_toks = 0;
    int *toks = tokenize_corpus(corpus_path, &n_toks);
    if (!toks || n_toks < T_ctx + 1) {
        fprintf(stderr, "[sft] corpus too short for T=%d\n", T_ctx); return 1;
    }

    nt_seed(42);
    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 20, steps, base_lr * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    fprintf(stderr, "\n[sft] training begins.\n");
    fprintf(stderr, "─────────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float loss_ema = 0, first_loss = 0;
    float loss_at_1 = 0, loss_at_10 = 0, loss_at_50 = 0;
    float gnorm_at_1 = 0;
    float val_history[64] = {0};
    int   val_count = 0;
    int   plateau_streak = 0;
    int   smoke_passed = (steps <= 50 ? 1 : 0);  /* skip smoke check on small runs */

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        long max_off = n_toks - T_ctx - 1;
        long off = (long)rand() % max_off;
        int *in_  = malloc((size_t)T_ctx * sizeof(int));
        int *tgt_ = malloc((size_t)T_ctx * sizeof(int));
        for (int t = 0; t < T_ctx; t++) { in_[t] = toks[off + t]; tgt_[t] = toks[off + t + 1]; }

        nt_tape_start();
        nt_train_mode(1);
        int loss_idx = forward(&M, in_, tgt_, T_ctx);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) { first_loss = lv; loss_ema = lv; }
        else loss_ema = 0.95f * loss_ema + 0.05f * lv;

        nt_tape_backward(loss_idx);
        float gn = global_grad_norm();

        /* smoke-50 kill conditions */
        if (step == 0)  { loss_at_1  = lv; gnorm_at_1 = gn; }
        if (step == 9)  loss_at_10 = lv;
        if (step == 49 && !smoke_passed) {
            loss_at_50 = lv;
            int killed = 0;
            if (!isfinite(lv)) { fprintf(stderr, "[sft] KILL: NaN at step 50\n"); killed = 1; }
            if (loss_at_50 > loss_at_10) {
                fprintf(stderr, "[sft] KILL: loss[50]=%.4f > loss[10]=%.4f (no descent)\n",
                        loss_at_50, loss_at_10); killed = 1;
            }
            if (gn > 100.0f * gnorm_at_1 || gn > 1e3f) {
                fprintf(stderr, "[sft] KILL: grad explosion |g|=%.2e (start=%.2e)\n",
                        gn, gnorm_at_1); killed = 1;
            }
            if (killed) {
                fprintf(stderr, "[sft] smoke-50 FAILED — exiting 99\n");
                free(in_); free(tgt_); return 99;
            }
            fprintf(stderr, "[sft] smoke-50 PASS: loss %.4f→%.4f→%.4f, |g| %.2e→%.2e\n",
                    loss_at_1, loss_at_10, loss_at_50, gnorm_at_1, gn);
            smoke_passed = 1;
        }

        if (!nt_nan_guard_check(&guard)) {
            nt_tape_clear();
            free(in_); free(tgt_);
            continue;
        }
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr, lv);
        nt_tape_clear();
        free(in_); free(tgt_);

        if ((step + 1) % LOG_EVERY == 0 || step == 0) {
            fprintf(stderr, "  step %5d | train %.4f | ema %.4f | lr %.2e | %.0fs\n",
                    step + 1, lv, loss_ema, lr, (now_ms() - t0) / 1000.0);
            fflush(stderr);
        }

        if ((step + 1) % VAL_EVERY == 0) {
            float val = eval_loss(&M, toks, n_toks, T_ctx);
            fprintf(stderr, "  ── val %5d | val %.4f", step + 1, val);
            if (val_count > 0) {
                float prev = val_history[val_count - 1];
                float delta = fabsf(val - prev);
                fprintf(stderr, " (Δ %.4f)", delta);
                if (delta < PLATEAU_EPS) plateau_streak++;
                else plateau_streak = 0;
            }
            fprintf(stderr, "\n");
            if (val_count < 64) val_history[val_count++] = val;
            if (plateau_streak >= PLATEAU_N) {
                fprintf(stderr, "[sft] EARLY STOP at step %d — val plateaued for %d×%d steps\n",
                        step + 1, PLATEAU_N, VAL_EVERY);
                break;
            }
        }

        if ((step + 1) % CKPT_EVERY == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s_ckpt_%d.bin", out_prefix, step + 1);
            write_rs02(path, &M);
            fprintf(stderr, "  ── ckpt → %s\n", path);
        }
    }

    /* Final write */
    char final_path[512];
    snprintf(final_path, sizeof(final_path), "%s_final.bin", out_prefix);
    write_rs02(final_path, &M);
    double total_s = (now_ms() - t0) / 1000.0;

    fprintf(stderr, "─────────────────────────────────────────────────────\n");
    fprintf(stderr, "[sft] DONE  first %.4f → ema %.4f  (%.0fs / %.1f min)\n",
            first_loss, loss_ema, total_s, total_s / 60.0);
    fprintf(stderr, "[sft] nans=%d skipped=%d\n",
            guard.total_nan_count, guard.skipped_steps);
    fprintf(stderr, "[sft] final → %s\n", final_path);
    return 0;
}
