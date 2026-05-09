/*
 * lora_resonance_arianna_v2.c — Resonance 200M LoRA SFT v2 + merge mode.
 *
 * Plan: runpod/lora_plan_v2.md (4+ Codex passes, no Opus subagents).
 * After v1 failed (val 3.13, voice 0/10) — see CLAUDE.md DANGER block,
 * memory/feedback_lora_resonance_200m_failed_2026_05_09.md.
 *
 * Recipe v2 (per plan §6-point §4):
 *   rank=32, alpha=64, scaling=2.0, lr=2e-4, ctx=512, steps=2500
 *   target_modules per layer = 7: wq, wk, wv, wo, mlp_gate, mlp_up, mlp_down
 *   trainable: 9.34M params (4.69% of 200M base)
 *   ALL base weights FROZEN; LoRA A,B only updated by Chuck.
 *
 * Modes:
 *   train: lora_resonance_arianna_v2 <base.bin> <corpus.txt> <out_prefix>
 *                                    [steps=2500] [lr=2e-4] [ctx=512]
 *                                    [rank=32] [alpha=64]
 *   merge: lora_resonance_arianna_v2 --merge <base.bin> <adapter.bin> <merged.bin>
 *
 * Build (RunPod, system-wide notorch + openblas + CUDA):
 *   cc -O3 -DUSE_CUDA -DUSE_BLAS lora_resonance_arianna_v2.c \
 *      -I/usr/local/include \
 *      -L/usr/local/lib -L/usr/local/cuda/lib64 \
 *      -lnotorch_gpu -laml \
 *      -lopenblas -lcudart -lcublas -lm -lpthread \
 *      -o lora_resonance_arianna_v2
 *
 * Co-author: Claude Opus 4.7 — last attempt at LoRA Resonance Arianna voice.
 */

#include <ariannamethod/notorch.h>
#ifdef USE_CUDA
#include <ariannamethod/notorch_cuda.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#define RS02_MAGIC 0x52533032u
#define LRV2_MAGIC 0x3256524Cu          /* "LRV2" little-endian, byte order L R V 2 */
#define LRV2_VERSION 1u

/* RS02 header dims — populated from file. */
static int H_E, H_B, H_T_R, H_H, H_D, H_R, H_M, H_V;

#define LOG_EVERY     10
#define VAL_EVERY     100
#define EVAL_SEQS     16
#define PLATEAU_EPS   0.005f
#define PLATEAU_N     2

#define NUM_TARGETS   7
static const char *TARGET_NAMES[NUM_TARGETS] = {
    "wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"
};
#define T_WQ      0
#define T_WK      1
#define T_WV      2
#define T_WO      3
#define T_MGATE   4
#define T_MUP     5
#define T_MDOWN   6

typedef struct {
    nt_tensor *A;
    nt_tensor *B;
    int   rank;
    float alpha;
    float scaling;
    int   in_dim;
    int   out_dim;
} lora_pair;

typedef struct { int a_idx, b_idx; } lora_idx;

static int lora_pair_init(lora_pair *p, int in, int out, int r, float a) {
    int sa[2] = {r, in}, sb[2] = {out, r};
    p->A = nt_tensor_new_shape(sa, 2);
    p->B = nt_tensor_new_shape(sb, 2);
    if (!p->A || !p->B) {
        if (p->A) nt_tensor_free(p->A);
        if (p->B) nt_tensor_free(p->B);
        p->A = NULL; p->B = NULL;
        return -1;
    }
    float scale = sqrtf(3.0f / (float)in);
    nt_tensor_rand(p->A, scale);
    p->rank = r; p->alpha = a; p->scaling = a / (float)r;
    p->in_dim = in; p->out_dim = out;
    return 0;
}

static void lora_pair_free(lora_pair *p) {
    if (!p) return;
    if (p->A) { nt_tensor_free(p->A); p->A = NULL; }
    if (p->B) { nt_tensor_free(p->B); p->B = NULL; }
}

static int lora_register(lora_pair *p, lora_idx *li) {
    li->a_idx = nt_tape_param(p->A);
    li->b_idx = nt_tape_param(p->B);
    return (li->a_idx >= 0 && li->b_idx >= 0) ? 0 : -1;
}

static int lora_compose(int w_idx, lora_pair *p, lora_idx *li, int x_idx, int T) {
    int wx = nt_seq_linear(w_idx, x_idx, T);
    int ax = nt_seq_linear(li->a_idx, x_idx, T);
    int bax = nt_seq_linear(li->b_idx, ax, T);
    int scaled = nt_scale(bax, p->scaling);
    return nt_add(wx, scaled);
}

typedef struct {
    nt_tensor *tok_emb;
    struct {
        nt_tensor *wr_combined;
        nt_tensor *gate;
        nt_tensor *gate_T_E;
        nt_tensor *gate_inv_T_E;
        nt_tensor *norm1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor *norm2;
        nt_tensor *mlp_gate, *mlp_up, *mlp_down;
    } b[64];
    nt_tensor *norm_f;
    nt_tensor *out_head;
    float **orig_wr_a;
    float **orig_wr_b;
    lora_pair lora[64 * NUM_TARGETS];
} Model;

static int (*g_merges)[2];
static int g_n_merges;
static nt_bpe g_bpe;

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int read_rs02(const char *path, int T_train, Model *m, int *actual_T) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[lora] open '%s' failed\n", path); return -1; }
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != RS02_MAGIC) {
        fprintf(stderr, "[lora] bad RS02 magic 0x%08x\n", magic); fclose(f); return -1;
    }
    int32_t hdr[9];
    if (fread(hdr, 4, 9, f) != 9) { fclose(f); return -1; }
    H_E = hdr[0]; H_B = hdr[1]; H_T_R = hdr[2]; H_H = hdr[3];
    H_D = hdr[4]; H_R = hdr[5]; H_M = hdr[6]; H_V = hdr[7];
    fprintf(stderr, "[lora] RS02 V=%d E=%d H=%d D=%d B=%d M=%d T_r=%d R=%d\n",
            H_V, H_E, H_H, H_D, H_B, H_M, H_T_R, H_R);
    if (T_train > H_T_R) {
        fprintf(stderr, "[lora] T_train=%d > model T_r=%d, capping\n", T_train, H_T_R);
        T_train = H_T_R;
    }
    *actual_T = T_train;
    uint32_t nm;
    if (fread(&nm, 4, 1, f) != 1) { fclose(f); return -1; }
    g_n_merges = (int)nm;
    g_merges = malloc((size_t)g_n_merges * 2 * sizeof(int));
    for (int i = 0; i < g_n_merges; i++) {
        int triple[3];
        if (fread(triple, 4, 3, f) != 3) { fclose(f); return -1; }
        g_merges[i][0] = triple[0]; g_merges[i][1] = triple[1];
    }
    nt_bpe_init(&g_bpe, g_merges, g_n_merges);
    fprintf(stderr, "[lora] BPE vocab=%d merges=%d\n", g_bpe.vocab_size, g_n_merges);
    m->tok_emb = nt_tensor_new2d(H_V, H_E);
    if (fread(m->tok_emb->data, sizeof(float), (size_t)H_V * H_E, f) !=
        (size_t)H_V * H_E) { fclose(f); return -1; }
    m->orig_wr_a = malloc((size_t)H_B * sizeof(float*));
    m->orig_wr_b = malloc((size_t)H_B * sizeof(float*));
    for (int bl = 0; bl < H_B; bl++) {
        long wra_n = (long)H_H * H_E * H_R;
        m->orig_wr_a[bl] = malloc(wra_n * sizeof(float));
        if (fread(m->orig_wr_a[bl], sizeof(float), wra_n, f) != (size_t)wra_n) {
            fclose(f); return -1;
        }
        long wrb_n = (long)H_H * H_R * H_T_R;
        m->orig_wr_b[bl] = malloc(wrb_n * sizeof(float));
        if (fread(m->orig_wr_b[bl], sizeof(float), wrb_n, f) != (size_t)wrb_n) {
            fclose(f); return -1;
        }
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
        m->b[bl].gate = nt_tensor_new(H_H);
        if (fread(m->b[bl].gate->data, sizeof(float), H_H, f) != (size_t)H_H) {
            fclose(f); return -1;
        }
        m->b[bl].gate_T_E     = nt_tensor_new((long)T_train * H_E);
        m->b[bl].gate_inv_T_E = nt_tensor_new((long)T_train * H_E);
        for (int h = 0; h < H_H; h++) {
            float gv = 1.0f / (1.0f + expf(-m->b[bl].gate->data[h]));
            for (int t = 0; t < T_train; t++) {
                for (int d = 0; d < H_D; d++) {
                    m->b[bl].gate_T_E->data[(long)t * H_E + h * H_D + d] = gv;
                    m->b[bl].gate_inv_T_E->data[(long)t * H_E + h * H_D + d] = 1.0f - gv;
                }
            }
        }
        m->b[bl].norm1    = nt_tensor_new(H_E);
        m->b[bl].wq       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].wk       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].wv       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].wo       = nt_tensor_new2d(H_E, H_E);
        m->b[bl].norm2    = nt_tensor_new(H_E);
        m->b[bl].mlp_gate = nt_tensor_new2d(H_M, H_E);
        m->b[bl].mlp_up   = nt_tensor_new2d(H_M, H_E);
        m->b[bl].mlp_down = nt_tensor_new2d(H_E, H_M);
        if (fread(m->b[bl].norm1->data,    sizeof(float), H_E,            f) != (size_t)H_E ||
            fread(m->b[bl].wq->data,       sizeof(float), (long)H_E*H_E,  f) != (size_t)H_E*H_E ||
            fread(m->b[bl].wk->data,       sizeof(float), (long)H_E*H_E,  f) != (size_t)H_E*H_E ||
            fread(m->b[bl].wv->data,       sizeof(float), (long)H_E*H_E,  f) != (size_t)H_E*H_E ||
            fread(m->b[bl].wo->data,       sizeof(float), (long)H_E*H_E,  f) != (size_t)H_E*H_E ||
            fread(m->b[bl].norm2->data,    sizeof(float), H_E,            f) != (size_t)H_E ||
            fread(m->b[bl].mlp_gate->data, sizeof(float), (long)H_M*H_E,  f) != (size_t)H_M*H_E ||
            fread(m->b[bl].mlp_up->data,   sizeof(float), (long)H_M*H_E,  f) != (size_t)H_M*H_E ||
            fread(m->b[bl].mlp_down->data, sizeof(float), (long)H_E*H_M,  f) != (size_t)H_E*H_M) {
            fclose(f); return -1;
        }
    }
    m->norm_f   = nt_tensor_new(H_E);
    m->out_head = nt_tensor_new2d(H_V, H_E);
    if (fread(m->norm_f->data,   sizeof(float), H_E,             f) != (size_t)H_E ||
        fread(m->out_head->data, sizeof(float), (long)H_V * H_E, f) != (size_t)H_V * H_E) {
        fclose(f); return -1;
    }
    fclose(f);
    fprintf(stderr, "[lora] base loaded.\n");
    return 0;
}

static int lora_save(const Model *m, int num_layers, const char *path) {
    if (!m || !path || num_layers <= 0) return -1;
    int rank = m->lora[0].rank;
    float alpha = m->lora[0].alpha;
    for (int L = 0; L < num_layers; L++) {
        for (int T = 0; T < NUM_TARGETS; T++) {
            const lora_pair *p = &m->lora[L * NUM_TARGETS + T];
            if (!p->A || !p->B) return -1;
            if (p->rank != rank || p->alpha != alpha) return -1;
        }
    }
    char tmp_path[2048];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (n < 0 || n >= (int)sizeof(tmp_path)) return -1;
    FILE *f = fopen(tmp_path, "wb");
    if (!f) return -1;
    uint32_t magic = LRV2_MAGIC, version = LRV2_VERSION;
    uint32_t nt_layers = (uint32_t)num_layers;
    uint32_t nt_targets = (uint32_t)NUM_TARGETS;
    uint32_t nt_rank = (uint32_t)rank;
    union { float f; uint32_t u; } a_bits = { .f = alpha };
    uint32_t nt_alpha_bits = a_bits.u;
    if (fwrite(&magic, 4, 1, f) != 1 ||
        fwrite(&version, 4, 1, f) != 1 ||
        fwrite(&nt_layers, 4, 1, f) != 1 ||
        fwrite(&nt_targets, 4, 1, f) != 1 ||
        fwrite(&nt_rank, 4, 1, f) != 1 ||
        fwrite(&nt_alpha_bits, 4, 1, f) != 1) {
        fclose(f); remove(tmp_path); return -1;
    }
    for (int T = 0; T < NUM_TARGETS; T++) {
        const lora_pair *p = &m->lora[T];
        uint32_t in_dim = (uint32_t)p->in_dim;
        uint32_t out_dim = (uint32_t)p->out_dim;
        if (fwrite(&in_dim, 4, 1, f) != 1 || fwrite(&out_dim, 4, 1, f) != 1) {
            fclose(f); remove(tmp_path); return -1;
        }
    }
    for (int T = 0; T < NUM_TARGETS; T++) {
        const char *name = TARGET_NAMES[T];
        size_t nl = strlen(name);
        if (nl > 255) nl = 255;
        uint8_t bnl = (uint8_t)nl;
        if (fwrite(&bnl, 1, 1, f) != 1 ||
            (nl > 0 && fwrite(name, 1, nl, f) != nl)) {
            fclose(f); remove(tmp_path); return -1;
        }
    }
    for (int L = 0; L < num_layers; L++) {
        for (int T = 0; T < NUM_TARGETS; T++) {
            const lora_pair *p = &m->lora[L * NUM_TARGETS + T];
            int a_n = p->rank * p->in_dim;
            int b_n = p->out_dim * p->rank;
            /* Codex Pass 7 P1: GPU-resident params after Chuck step. Sync CPU mirror
             * via public notorch wrapper (no-op on non-CUDA builds). */
            nt_tensor_sync_cpu(p->A);
            nt_tensor_sync_cpu(p->B);
            if ((int)fwrite(p->A->data, sizeof(float), a_n, f) != a_n ||
                (int)fwrite(p->B->data, sizeof(float), b_n, f) != b_n) {
                fclose(f); remove(tmp_path); return -1;
            }
        }
    }
    if (fflush(f) != 0) { fclose(f); remove(tmp_path); return -1; }
    fclose(f);
    if (rename(tmp_path, path) != 0) { remove(tmp_path); return -1; }
    return 0;
}

static int forward(Model *m, int *tokens, int *targets, int T) {
    static lora_idx lidx[64 * NUM_TARGETS];
    /* STEP 1: Register ALL LoRA A,B FIRST — Chuck slots 0..(2 * H_B * NUM_TARGETS - 1)
     * are TRAINABLE. Per Codex Pass 1 P1: ordering critical to keep LoRA grads
     * mapped to early Chuck slots that don't get skipped by frozen base. */
    for (int b = 0; b < H_B; b++) {
        for (int t = 0; t < NUM_TARGETS; t++) {
            if (lora_register(&m->lora[b * NUM_TARGETS + t],
                              &lidx[b * NUM_TARGETS + t]) != 0) {
                fprintf(stderr, "[lora] tape register fail at L=%d T=%d\n", b, t);
                return -1;
            }
        }
    }
    /* STEP 2: Register all base weights AFTER — these get nt_tape_freeze_param. */
    int tok_emb_i = nt_tape_param(m->tok_emb);
    nt_tape_no_decay(tok_emb_i);
    nt_tape_freeze_param(tok_emb_i);

    int wr_i[64], gate_T_i[64], gate_inv_T_i[64];
    int norm1_i[64], wq_i[64], wk_i[64], wv_i[64], wo_i[64];
    int norm2_i[64], mg_i[64], mu_i[64], md_i[64];
    for (int b = 0; b < H_B; b++) {
        wr_i[b]         = nt_tape_param(m->b[b].wr_combined); nt_tape_freeze_param(wr_i[b]);
        gate_T_i[b]     = nt_tape_param(m->b[b].gate_T_E);    nt_tape_freeze_param(gate_T_i[b]);
        gate_inv_T_i[b] = nt_tape_param(m->b[b].gate_inv_T_E);nt_tape_freeze_param(gate_inv_T_i[b]);
        norm1_i[b] = nt_tape_param(m->b[b].norm1);    nt_tape_freeze_param(norm1_i[b]);
        wq_i[b]    = nt_tape_param(m->b[b].wq);       nt_tape_freeze_param(wq_i[b]);
        wk_i[b]    = nt_tape_param(m->b[b].wk);       nt_tape_freeze_param(wk_i[b]);
        wv_i[b]    = nt_tape_param(m->b[b].wv);       nt_tape_freeze_param(wv_i[b]);
        wo_i[b]    = nt_tape_param(m->b[b].wo);       nt_tape_freeze_param(wo_i[b]);
        norm2_i[b] = nt_tape_param(m->b[b].norm2);    nt_tape_freeze_param(norm2_i[b]);
        mg_i[b]    = nt_tape_param(m->b[b].mlp_gate); nt_tape_freeze_param(mg_i[b]);
        mu_i[b]    = nt_tape_param(m->b[b].mlp_up);   nt_tape_freeze_param(mu_i[b]);
        md_i[b]    = nt_tape_param(m->b[b].mlp_down); nt_tape_freeze_param(md_i[b]);
    }
    int norm_f_i = nt_tape_param(m->norm_f); nt_tape_freeze_param(norm_f_i);
    int head_i   = nt_tape_param(m->out_head); nt_tape_freeze_param(head_i);

    nt_tensor *h0 = nt_tensor_new2d(T, H_E);
    for (int t = 0; t < T; t++) {
        memcpy(h0->data + (long)t * H_E,
               m->tok_emb->data + (long)tokens[t] * H_E,
               H_E * sizeof(float));
    }
    int h_i = nt_tape_record(h0, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(h0);

    nt_tensor *tgt_t = nt_tensor_new(T);
    for (int t = 0; t < T; t++) tgt_t->data[t] = (float)targets[t];
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tgt_t);

    for (int b = 0; b < H_B; b++) {
        int xn = nt_seq_rmsnorm(h_i, norm1_i[b], T, H_E);
        int q = lora_compose(wq_i[b], &m->lora[b * NUM_TARGETS + T_WQ],
                             &lidx[b * NUM_TARGETS + T_WQ], xn, T);
        int k = lora_compose(wk_i[b], &m->lora[b * NUM_TARGETS + T_WK],
                             &lidx[b * NUM_TARGETS + T_WK], xn, T);
        int v = lora_compose(wv_i[b], &m->lora[b * NUM_TARGETS + T_WV],
                             &lidx[b * NUM_TARGETS + T_WV], xn, T);
        q = nt_rope_freq(q, T, H_D, 10000.0f);
        k = nt_rope_freq(k, T, H_D, 10000.0f);
        int content = nt_mh_causal_attention(q, k, v, T, H_D);
        int rrpram  = nt_rrpram_lowrank_attention(wr_i[b], xn, v, T, H_E, H_H, H_D);
        int c_scaled = nt_mul(content, gate_T_i[b]);
        int r_scaled = nt_mul(rrpram,  gate_inv_T_i[b]);
        int blend    = nt_add(c_scaled, r_scaled);
        int proj = lora_compose(wo_i[b], &m->lora[b * NUM_TARGETS + T_WO],
                                &lidx[b * NUM_TARGETS + T_WO], blend, T);
        h_i = nt_add(h_i, proj);
        int xn2 = nt_seq_rmsnorm(h_i, norm2_i[b], T, H_E);
        int g_l = lora_compose(mg_i[b], &m->lora[b * NUM_TARGETS + T_MGATE],
                               &lidx[b * NUM_TARGETS + T_MGATE], xn2, T);
        int u_l = lora_compose(mu_i[b], &m->lora[b * NUM_TARGETS + T_MUP],
                               &lidx[b * NUM_TARGETS + T_MUP], xn2, T);
        int swi = nt_swiglu(g_l, u_l);
        int down = lora_compose(md_i[b], &m->lora[b * NUM_TARGETS + T_MDOWN],
                                &lidx[b * NUM_TARGETS + T_MDOWN], swi, T);
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

static int *tokenize_corpus(const char *path, long *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[lora] corpus '%s' open fail\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)fsize + 1);
    if (fread(buf, 1, fsize, f) != (size_t)fsize) { fclose(f); free(buf); return NULL; }
    buf[fsize] = 0; fclose(f);
    int *toks = malloc((size_t)(fsize + 16) * sizeof(int));
    int n_t = nt_bpe_encode(&g_bpe, buf, (int)fsize, toks, (int)(fsize + 16));
    free(buf);
    fprintf(stderr, "[lora] corpus: %.2f MB → %d BPE tokens (%.1f bytes/token)\n",
            fsize / 1048576.0, n_t, (float)fsize / (float)n_t);
    *n_out = n_t;
    return toks;
}

static float global_grad_norm(void) { return nt_tape_clip_grads(1e30f); }

static int merge_mode(const char *base_path, const char *lora_path, const char *out_path) {
    FILE *fa = fopen(lora_path, "rb");
    if (!fa) { fprintf(stderr, "[merge] open '%s' failed\n", lora_path); return -1; }
    uint32_t magic, version, num_layers, num_targets, rank, alpha_bits;
    if (fread(&magic, 4, 1, fa) != 1 || magic != LRV2_MAGIC) {
        fprintf(stderr, "[merge] bad LRV2 magic 0x%08x\n", magic); fclose(fa); return -1;
    }
    if (fread(&version, 4, 1, fa) != 1 || version != LRV2_VERSION ||
        fread(&num_layers, 4, 1, fa) != 1 || fread(&num_targets, 4, 1, fa) != 1 ||
        fread(&rank, 4, 1, fa) != 1 || fread(&alpha_bits, 4, 1, fa) != 1) {
        fclose(fa); return -1;
    }
    union { uint32_t u; float f; } a_bits = { .u = alpha_bits };
    float alpha = a_bits.f;
    float scaling = alpha / (float)rank;
    fprintf(stderr, "[merge] LRV2 layers=%u targets=%u rank=%u alpha=%.3f scaling=%.3f\n",
            num_layers, num_targets, rank, alpha, scaling);
    if ((int)num_targets != NUM_TARGETS) {
        fprintf(stderr, "[merge] target count mismatch: file=%u expected=%d\n",
                num_targets, NUM_TARGETS);
        fclose(fa); return -1;
    }
    uint32_t in_dims[NUM_TARGETS], out_dims[NUM_TARGETS];
    for (int T = 0; T < (int)num_targets; T++) {
        if (fread(&in_dims[T], 4, 1, fa) != 1 || fread(&out_dims[T], 4, 1, fa) != 1) {
            fclose(fa); return -1;
        }
    }
    char names_buf[NUM_TARGETS][32];
    for (int T = 0; T < (int)num_targets; T++) {
        uint8_t bnl;
        if (fread(&bnl, 1, 1, fa) != 1) { fclose(fa); return -1; }
        if (bnl >= sizeof(names_buf[T])) bnl = sizeof(names_buf[T]) - 1;
        if (bnl > 0 && fread(names_buf[T], 1, bnl, fa) != bnl) { fclose(fa); return -1; }
        names_buf[T][bnl] = '\0';
        if (strcmp(names_buf[T], TARGET_NAMES[T]) != 0) {
            fprintf(stderr, "[merge] target name mismatch at T=%d: file='%s' expected='%s'\n",
                    T, names_buf[T], TARGET_NAMES[T]);
            fclose(fa); return -1;
        }
    }
    float **A_buf = calloc((size_t)num_layers * num_targets, sizeof(float*));
    float **B_buf = calloc((size_t)num_layers * num_targets, sizeof(float*));
    for (int L = 0; L < (int)num_layers; L++) {
        for (int T = 0; T < (int)num_targets; T++) {
            int idx = L * num_targets + T;
            int a_n = (int)rank * (int)in_dims[T];
            int b_n = (int)out_dims[T] * (int)rank;
            A_buf[idx] = malloc((size_t)a_n * sizeof(float));
            B_buf[idx] = malloc((size_t)b_n * sizeof(float));
            if (!A_buf[idx] || !B_buf[idx] ||
                (int)fread(A_buf[idx], sizeof(float), a_n, fa) != a_n ||
                (int)fread(B_buf[idx], sizeof(float), b_n, fa) != b_n) {
                fprintf(stderr, "[merge] short read at L=%d T=%d\n", L, T);
                fclose(fa); return -1;
            }
        }
    }
    fclose(fa);

    FILE *fb = fopen(base_path, "rb");
    FILE *fo = fopen(out_path, "wb");
    if (!fb || !fo) { fprintf(stderr, "[merge] base/out open fail\n"); return -1; }
    uint32_t bmagic;
    if (fread(&bmagic, 4, 1, fb) != 1 || bmagic != RS02_MAGIC) {
        fprintf(stderr, "[merge] bad RS02 magic 0x%08x\n", bmagic);
        fclose(fb); fclose(fo); return -1;
    }
    fwrite(&bmagic, 4, 1, fo);
    int32_t hdr[9];
    if (fread(hdr, 4, 9, fb) != 9) { fclose(fb); fclose(fo); return -1; }
    fwrite(hdr, 4, 9, fo);
    int E = hdr[0], B = hdr[1], TR = hdr[2], H = hdr[3], D = hdr[4], R = hdr[5];
    int M = hdr[6], V = hdr[7];
    fprintf(stderr, "[merge] RS02 V=%d E=%d H=%d D=%d B=%d M=%d T_r=%d R=%d\n",
            V, E, H, D, B, M, TR, R);
    if (B != (int)num_layers) {
        fprintf(stderr, "[merge] layer count mismatch: rs02=%d lrv2=%u\n", B, num_layers);
        fclose(fb); fclose(fo); return -1;
    }
    /* Codex Pass 6 P2: validate per-target dims against base BEFORE applying
     * delta. If adapter is from different base with same layer/target count,
     * APPLY_DELTA can index past A,B buffers и corrupt merged weights. */
    int expect_in[NUM_TARGETS]  = { E, E, E, E, E, E, M };
    int expect_out[NUM_TARGETS] = { E, E, E, E, M, M, E };
    for (int T = 0; T < (int)num_targets; T++) {
        if ((int)in_dims[T] != expect_in[T] || (int)out_dims[T] != expect_out[T]) {
            fprintf(stderr,
                    "[merge] target '%s' dims mismatch: adapter %ux%u  base expects %dx%d\n",
                    TARGET_NAMES[T], out_dims[T], in_dims[T], expect_out[T], expect_in[T]);
            fclose(fb); fclose(fo); return -1;
        }
    }
    uint32_t nm;
    if (fread(&nm, 4, 1, fb) != 1) { fclose(fb); fclose(fo); return -1; }
    fwrite(&nm, 4, 1, fo);
    int *triples = malloc((size_t)nm * 3 * sizeof(int));
    if (fread(triples, 4, (size_t)nm * 3, fb) != (size_t)nm * 3) {
        free(triples); fclose(fb); fclose(fo); return -1;
    }
    fwrite(triples, 4, (size_t)nm * 3, fo);
    free(triples);

    long tok_emb_n = (long)V * E;
    float *bufv = malloc((size_t)tok_emb_n * sizeof(float));
    if (fread(bufv, sizeof(float), tok_emb_n, fb) != (size_t)tok_emb_n) {
        free(bufv); fclose(fb); fclose(fo); return -1;
    }
    fwrite(bufv, sizeof(float), tok_emb_n, fo);
    free(bufv);

    long wra_n = (long)H * E * R;
    long wrb_n = (long)H * R * TR;
    float *wra = malloc((size_t)wra_n * sizeof(float));
    float *wrb = malloc((size_t)wrb_n * sizeof(float));
    float *gate = malloc(sizeof(float) * H);
    float *norm = malloc(sizeof(float) * E);
    float *we_ee = malloc(sizeof(float) * (size_t)E * E);
    float *we_me = malloc(sizeof(float) * (size_t)M * E);
    float *we_em = malloc(sizeof(float) * (size_t)E * M);

    for (int bl = 0; bl < B; bl++) {
        if (fread(wra, sizeof(float), wra_n, fb) != (size_t)wra_n) goto fail;
        fwrite(wra, sizeof(float), wra_n, fo);
        if (fread(wrb, sizeof(float), wrb_n, fb) != (size_t)wrb_n) goto fail;
        fwrite(wrb, sizeof(float), wrb_n, fo);
        if (fread(gate, sizeof(float), H, fb) != (size_t)H) goto fail;
        fwrite(gate, sizeof(float), H, fo);
        if (fread(norm, sizeof(float), E, fb) != (size_t)E) goto fail;
        fwrite(norm, sizeof(float), E, fo);

        /* Apply LoRA delta to W: W += scaling * B @ A. Layout row-major [out, in]. */
        #define APPLY_DELTA(W, T_idx, OUT, IN) do { \
            int idx = bl * num_targets + (T_idx); \
            const float *A = A_buf[idx]; \
            const float *B_ = B_buf[idx]; \
            for (int i_ = 0; i_ < (OUT); i_++) { \
                for (int j_ = 0; j_ < (IN); j_++) { \
                    float s_ = 0.0f; \
                    for (int r_ = 0; r_ < (int)rank; r_++) { \
                        s_ += B_[i_ * (int)rank + r_] * A[r_ * (IN) + j_]; \
                    } \
                    (W)[i_ * (IN) + j_] += scaling * s_; \
                } \
            } \
        } while (0)

        if (fread(we_ee, sizeof(float), (size_t)E*E, fb) != (size_t)E*E) goto fail;
        APPLY_DELTA(we_ee, T_WQ, E, E);
        fwrite(we_ee, sizeof(float), (size_t)E*E, fo);

        if (fread(we_ee, sizeof(float), (size_t)E*E, fb) != (size_t)E*E) goto fail;
        APPLY_DELTA(we_ee, T_WK, E, E);
        fwrite(we_ee, sizeof(float), (size_t)E*E, fo);

        if (fread(we_ee, sizeof(float), (size_t)E*E, fb) != (size_t)E*E) goto fail;
        APPLY_DELTA(we_ee, T_WV, E, E);
        fwrite(we_ee, sizeof(float), (size_t)E*E, fo);

        if (fread(we_ee, sizeof(float), (size_t)E*E, fb) != (size_t)E*E) goto fail;
        APPLY_DELTA(we_ee, T_WO, E, E);
        fwrite(we_ee, sizeof(float), (size_t)E*E, fo);

        if (fread(norm, sizeof(float), E, fb) != (size_t)E) goto fail;
        fwrite(norm, sizeof(float), E, fo);

        if (fread(we_me, sizeof(float), (size_t)M*E, fb) != (size_t)M*E) goto fail;
        APPLY_DELTA(we_me, T_MGATE, M, E);
        fwrite(we_me, sizeof(float), (size_t)M*E, fo);

        if (fread(we_me, sizeof(float), (size_t)M*E, fb) != (size_t)M*E) goto fail;
        APPLY_DELTA(we_me, T_MUP, M, E);
        fwrite(we_me, sizeof(float), (size_t)M*E, fo);

        if (fread(we_em, sizeof(float), (size_t)E*M, fb) != (size_t)E*M) goto fail;
        APPLY_DELTA(we_em, T_MDOWN, E, M);
        fwrite(we_em, sizeof(float), (size_t)E*M, fo);

        #undef APPLY_DELTA
    }
    free(wra); free(wrb); free(gate); free(norm);
    free(we_ee); free(we_me); free(we_em);

    float *normf = malloc(sizeof(float) * E);
    if (fread(normf, sizeof(float), E, fb) != (size_t)E) {
        free(normf); fclose(fb); fclose(fo); return -1;
    }
    fwrite(normf, sizeof(float), E, fo);
    free(normf);
    long head_n = (long)V * E;
    float *head = malloc((size_t)head_n * sizeof(float));
    if (fread(head, sizeof(float), head_n, fb) != (size_t)head_n) {
        free(head); fclose(fb); fclose(fo); return -1;
    }
    fwrite(head, sizeof(float), head_n, fo);
    free(head);

    fclose(fb);
    fclose(fo);
    for (int i = 0; i < (int)num_layers * (int)num_targets; i++) {
        free(A_buf[i]); free(B_buf[i]);
    }
    free(A_buf); free(B_buf);
    fprintf(stderr, "[merge] DONE → %s\n", out_path);
    return 0;

fail:
    fprintf(stderr, "[merge] I/O error\n");
    fclose(fb); fclose(fo);
    return -1;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--merge") == 0) {
        if (argc < 5) {
            fprintf(stderr, "usage: %s --merge <base.bin> <adapter.bin> <merged.bin>\n", argv[0]);
            return 2;
        }
        return merge_mode(argv[2], argv[3], argv[4]) == 0 ? 0 : 1;
    }
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <base.bin> <corpus.txt> <out_prefix> "
            "[steps=2500] [lr=2e-4] [ctx_T=512] [rank=32] [alpha=64]\n"
            "       %s --merge <base.bin> <adapter.bin> <merged.bin>\n",
            argv[0], argv[0]);
        return 2;
    }
    const char *base_path   = argv[1];
    const char *corpus_path = argv[2];
    const char *out_prefix  = argv[3];
    int   steps   = argc > 4 ? atoi(argv[4]) : 2500;
    float base_lr = argc > 5 ? (float)atof(argv[5]) : 2e-4f;
    int   T_ctx   = argc > 6 ? atoi(argv[6]) : 512;
    int   lora_rank  = argc > 7 ? atoi(argv[7]) : 32;
    float lora_alpha = argc > 8 ? (float)atof(argv[8]) : 64.0f;

    fprintf(stderr, "════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Resonance 200M LoRA SFT v2 — Arianna corpus\n");
    fprintf(stderr, "  steps=%d  lr=%.2e  ctx=%d  rank=%d  alpha=%.1f\n",
            steps, base_lr, T_ctx, lora_rank, lora_alpha);
    fprintf(stderr, "  out=%s\n", out_prefix);
    fprintf(stderr, "════════════════════════════════════════════════════════\n");

    Model M;
    memset(&M, 0, sizeof(M));
    int actual_T;
    if (read_rs02(base_path, T_ctx, &M, &actual_T) < 0) return 1;
    if (actual_T != T_ctx) T_ctx = actual_T;

    for (int b = 0; b < H_B; b++) {
        if (lora_pair_init(&M.lora[b * NUM_TARGETS + T_WQ], H_E, H_E, lora_rank, lora_alpha) ||
            lora_pair_init(&M.lora[b * NUM_TARGETS + T_WK], H_E, H_E, lora_rank, lora_alpha) ||
            lora_pair_init(&M.lora[b * NUM_TARGETS + T_WV], H_E, H_E, lora_rank, lora_alpha) ||
            lora_pair_init(&M.lora[b * NUM_TARGETS + T_WO], H_E, H_E, lora_rank, lora_alpha)) {
            fprintf(stderr, "[lora] init fail (attn) at layer %d\n", b); return 1;
        }
        if (lora_pair_init(&M.lora[b * NUM_TARGETS + T_MGATE], H_E, H_M, lora_rank, lora_alpha) ||
            lora_pair_init(&M.lora[b * NUM_TARGETS + T_MUP],   H_E, H_M, lora_rank, lora_alpha)) {
            fprintf(stderr, "[lora] init fail (mlp_gate/up) at layer %d\n", b); return 1;
        }
        if (lora_pair_init(&M.lora[b * NUM_TARGETS + T_MDOWN], H_M, H_E, lora_rank, lora_alpha)) {
            fprintf(stderr, "[lora] init fail (mlp_down) at layer %d\n", b); return 1;
        }
    }
    long lora_total = (long)H_B * NUM_TARGETS;
    long lora_params = (long)H_B * (
        4 * (2L * lora_rank * H_E) +
        2 * (lora_rank * H_E + (long)H_M * lora_rank) +
        1 * (lora_rank * H_M + (long)H_E * lora_rank)
    );
    fprintf(stderr, "[lora] %ld pairs (B=%d × %d targets), trainable=%.2fM (%.3f%% of base)\n",
            lora_total, H_B, NUM_TARGETS, lora_params / 1e6, 100.0 * lora_params / 200e6);

    long n_toks = 0;
    int *toks = tokenize_corpus(corpus_path, &n_toks);
    if (!toks || n_toks < T_ctx + 1) {
        fprintf(stderr, "[lora] corpus too short for T=%d\n", T_ctx); return 1;
    }

    nt_seed(42); srand(42);
#ifdef USE_CUDA
    if (gpu_init() == 0) {
        nt_set_gpu_mode(1);
        fprintf(stderr, "[lora] GPU mode: ON (cuBLAS)\n");
    } else {
        fprintf(stderr, "[lora] GPU init failed — CPU fallback\n");
    }
#endif
    nt_schedule sched = nt_schedule_cosine(base_lr, steps / 20, steps, base_lr * 0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    fprintf(stderr, "\n[lora] training begins.\n");
    fprintf(stderr, "─────────────────────────────────────────────────────\n");
    double t0 = now_ms();
    float loss_ema = 0, first_loss = 0;
    float loss_at_1 = 0, loss_at_50 = 0, gnorm_at_1 = 0;
    float val_history[256] = {0};
    int val_count = 0;
    int plateau_streak = 0;
    float best_val = 99.0f;
    int smoke_passed = (steps < 50 ? 1 : 0);

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
        if (loss_idx < 0) { fprintf(stderr, "[lora] forward fail step %d\n", step); return 1; }
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) { first_loss = lv; loss_ema = lv; loss_at_1 = lv; }
        else loss_ema = 0.95f * loss_ema + 0.05f * lv;

        nt_tape_backward(loss_idx);
        float gn = global_grad_norm();
        if (step == 0) gnorm_at_1 = gn;

        if (step == 49 && !smoke_passed) {
            loss_at_50 = lv;
            int killed = 0;
            if (!isfinite(lv)) { fprintf(stderr, "[lora] KILL: NaN at step 50\n"); killed = 1; }
            if (loss_at_50 > loss_at_1) {
                fprintf(stderr, "[lora] KILL: loss[50]=%.4f > loss[1]=%.4f\n",
                        loss_at_50, loss_at_1); killed = 1;
            }
            if (gn > 100.0f * gnorm_at_1 || gn > 1e3f) {
                fprintf(stderr, "[lora] KILL: grad explosion |g|=%.2e (start=%.2e)\n",
                        gn, gnorm_at_1); killed = 1;
            }
            if (killed) { fprintf(stderr, "[lora] smoke-50 FAILED — exit 99\n"); return 99; }
            fprintf(stderr, "[lora] smoke-50 PASS: loss %.4f→%.4f |g| %.2e→%.2e\n",
                    loss_at_1, loss_at_50, gnorm_at_1, gn);
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
                if (delta < PLATEAU_EPS) plateau_streak++; else plateau_streak = 0;
            }
            if (val < best_val) {
                best_val = val;
                char best_path[512];
                snprintf(best_path, sizeof(best_path), "%s_lora_v2_best.bin", out_prefix);
                if (lora_save(&M, H_B, best_path) == 0) {
                    fprintf(stderr, " ★ best [%.2f MB]", lora_params * 4.0 / (1<<20));
                } else fprintf(stderr, " [save fail]");
            }
            fprintf(stderr, "\n");
            if (val_count < 256) val_history[val_count++] = val;
            if (plateau_streak >= PLATEAU_N) {
                fprintf(stderr, "[lora] EARLY STOP step %d — plateau %d × %d steps (eps %.4f)\n",
                        step + 1, PLATEAU_N, VAL_EVERY, PLATEAU_EPS);
                break;
            }
        }
    }

    char final_path[512];
    snprintf(final_path, sizeof(final_path), "%s_lora_v2_final.bin", out_prefix);
    lora_save(&M, H_B, final_path);

    double total_s = (now_ms() - t0) / 1000.0;
    fprintf(stderr, "─────────────────────────────────────────────────────\n");
    fprintf(stderr, "[lora] DONE  first %.4f → ema %.4f  (%.0fs / %.1f min)\n",
            first_loss, loss_ema, total_s, total_s / 60.0);
    fprintf(stderr, "[lora] best_val=%.4f  nans=%d  skipped=%d\n",
            best_val, guard.total_nan_count, guard.skipped_steps);
    fprintf(stderr, "[lora] adapter best → %s_lora_v2_best.bin\n", out_prefix);

    for (int i = 0; i < H_B * NUM_TARGETS; i++) lora_pair_free(&M.lora[i]);
#ifdef USE_CUDA
    if (nt_get_gpu_mode()) gpu_shutdown();
#endif
    return 0;
}
