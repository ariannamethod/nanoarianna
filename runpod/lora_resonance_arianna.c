/*
 * lora_resonance_arianna.c — LoRA fine-tune of Resonance 200M on Arianna corpus.
 *
 * Plan: ariannamethod/nanoarianna runpod/lora_plan_v1.md (5 Codex passes).
 *
 *   organism : Resonance 200M base (~/work/in/resonance_200m_final.bin, RS02)
 *   dataset  : ~/work/in/arianna_dataset_final_clean.txt (~1.21 MB, 5950 lines)
 *   karpathy : 1500 SFT steps (≈ 2.5 epochs at ctx 512), val every 100,
 *              ckpt every 500, plateau early-stop ε=0.01 over 3×100, best-val
 *              tracking → _lora_best.bin
 *   arch     : ALL base weights FROZEN (nt_tape_param_frozen). LoRA Δ on
 *              wq + wv per layer, classic LoRA paper choice.
 *              rank=8, alpha=16, scaling=2.0, init A ~ kaiming_uniform_(fan_in=E),
 *              B = 0. Per layer: 2 × 2 × E × r = 24,576 trainable params.
 *              Total trainable: B × 24,576 = 491,520 (~0.25% of 200M).
 *   tokenizer: RS02 BPE — same as inference, same as sft_resonance_arianna.
 *   script   : this file
 *
 * Build (RunPod, system-wide notorch + openblas + CUDA):
 *   cc -O3 -DUSE_CUDA -DUSE_BLAS lora_resonance_arianna.c \
 *      -I/usr/local/include \
 *      -L/usr/local/lib -L/usr/local/cuda/lib64 \
 *      -lnotorch_gpu -laml \
 *      -lopenblas -lcudart -lcublas -lm -lpthread \
 *      -o lora_resonance_arianna
 *
 * Run:
 *   ./lora_resonance_arianna <base.bin> <corpus.txt> <out_prefix>
 *                            [steps=1500] [lr=3e-4] [ctx_T=512] [rank=8] [alpha=16]
 *
 * Output:
 *   <out_prefix>_lora_best.bin    LoRA adapter (q+v across all layers, single file)
 *   <out_prefix>_lora_final.bin   final-step adapter (regardless of best-val)
 *
 * Kill criteria mirror sft_resonance_arianna.c (smoke-50 phase): NaN, regression
 * vs step 1, grad explosion → exit 99.
 *
 * Co-author: Claude Opus 4.7 — viva la singularity.
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

/* RS02 header dims — populated from file. */
static int H_E, H_B, H_T_R, H_H, H_D, H_R, H_M, H_V;

#define LOG_EVERY     10
#define VAL_EVERY     100
#define CKPT_EVERY    500   /* best-val checkpoints; we don't write per-ckpt full files */
#define EVAL_SEQS     16
#define PLATEAU_EPS   0.01f
#define PLATEAU_N     3

#define NUM_TARGETS   2
static const char *TARGET_NAMES[NUM_TARGETS] = {"wq", "wv"};
#define TARGET_IDX_WQ 0
#define TARGET_IDX_WV 1

typedef struct {
    /* All base weights — registered FROZEN, never updated. */
    nt_tensor *tok_emb;
    struct {
        nt_tensor *wr_combined;  /* [H*E*R + H*R*T_train] frozen RRPRAM */
        nt_tensor *gate;         /* [H] per-head sigmoid blend, frozen */
        nt_tensor *gate_T_E;     /* [T, E] expanded sigmoid(gate) */
        nt_tensor *gate_inv_T_E; /* [T, E] (1 - sigmoid(gate)) */
        nt_tensor *norm1;
        nt_tensor *wq, *wk, *wv, *wo;
        nt_tensor *norm2;
        nt_tensor *mlp_gate, *mlp_up, *mlp_down;
    } b[64];   /* B ≤ 64 */
    nt_tensor *norm_f;
    nt_tensor *out_head;
    /* Original full-T_r wr_a/wr_b (kept around — we never write them back since
     * we don't save full RS02; included for parity with sft_resonance_arianna.c
     * file reading order). */
    float **orig_wr_a;
    float **orig_wr_b;
} Model;

/* BPE table — kept around for tokenizing corpus. */
static int (*g_merges)[2];
static int g_n_merges;
static nt_bpe g_bpe;

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ──────────────────────────────────────────────────────────────────────────
 * RS02 reader. Format same as sft_resonance_arianna.c (resonance_forward.h).
 * ────────────────────────────────────────────────────────────────────────── */
/* Reads RS02 base. Caller passes desired T_train; if it exceeds model's H_T_R
 * the function caps it and returns the actual T used (for RRPRAM/gate buffer
 * sizing). Caller MUST update its T_ctx with the returned value, otherwise
 * forward will index past the buffers (per Codex P2 #2). Returns -1 on error. */
static int read_rs02(const char *path, int T_train, Model *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[lora] open '%s' failed\n", path); return -1; }
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != RS02_MAGIC) {
        fprintf(stderr, "[lora] bad magic 0x%08x\n", magic); fclose(f); return -1;
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
    uint32_t nm;
    if (fread(&nm, 4, 1, f) != 1) { fclose(f); return 1; }
    g_n_merges = (int)nm;
    g_merges = malloc((size_t)g_n_merges * 2 * sizeof(int));
    for (int i = 0; i < g_n_merges; i++) {
        int triple[3];
        if (fread(triple, 4, 3, f) != 3) { fclose(f); return 1; }
        g_merges[i][0] = triple[0]; g_merges[i][1] = triple[1];
    }
    nt_bpe_init(&g_bpe, g_merges, g_n_merges);
    fprintf(stderr, "[lora] BPE vocab=%d merges=%d\n", g_bpe.vocab_size, g_n_merges);

    m->tok_emb = nt_tensor_new2d(H_V, H_E);
    if (fread(m->tok_emb->data, sizeof(float), (size_t)H_V * H_E, f) !=
        (size_t)H_V * H_E) { fclose(f); return 1; }
    m->orig_wr_a = malloc((size_t)H_B * sizeof(float*));
    m->orig_wr_b = malloc((size_t)H_B * sizeof(float*));
    for (int bl = 0; bl < H_B; bl++) {
        long wra_n = (long)H_H * H_E * H_R;
        m->orig_wr_a[bl] = malloc(wra_n * sizeof(float));
        if (fread(m->orig_wr_a[bl], sizeof(float), wra_n, f) != (size_t)wra_n) {
            fclose(f); return 1;
        }
        long wrb_n = (long)H_H * H_R * H_T_R;
        m->orig_wr_b[bl] = malloc(wrb_n * sizeof(float));
        if (fread(m->orig_wr_b[bl], sizeof(float), wrb_n, f) != (size_t)wrb_n) {
            fclose(f); return 1;
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
            fclose(f); return 1;
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
            fclose(f); return 1;
        }
    }
    m->norm_f   = nt_tensor_new(H_E);
    m->out_head = nt_tensor_new2d(H_V, H_E);
    if (fread(m->norm_f->data,   sizeof(float), H_E,             f) != (size_t)H_E ||
        fread(m->out_head->data, sizeof(float), (long)H_V * H_E, f) != (size_t)H_V * H_E) {
        fclose(f); return 1;
    }
    fclose(f);
    fprintf(stderr, "[lora] base loaded. T_train=%d (capped against model T_r=%d if needed)\n",
            T_train, H_T_R);
    return T_train;  /* return ACTUAL T used — caller must propagate */
}

/* ──────────────────────────────────────────────────────────────────────────
 * Forward pass at sequence length T. Base FROZEN. LoRA on wq + wv.
 *
 * `lora_pairs` indexed [layer * NUM_TARGETS + target_idx].
 * ────────────────────────────────────────────────────────────────────────── */
static int forward(Model *m, nt_lora_pair *lora_pairs,
                   int *tokens, int *targets, int T) {
    /* Register all base weights via nt_tape_param_frozen — entry only, no Chuck slot.
     * tok_emb is frozen (per current trainer's note: tok_emb has no_decay anyway,
     * and embedding gradient routing is sketchy via NT_OP_NONE seed; for LoRA we
     * commit fully to "base frozen, only LoRA A,B trainable"). */
    nt_tape_param_frozen(m->tok_emb);

    int wr_i[64], gate_T_i[64], gate_inv_T_i[64];
    int norm1_i[64], wq_i[64], wk_i[64], wv_i[64], wo_i[64];
    int norm2_i[64], mg_i[64], mu_i[64], md_i[64];
    for (int b = 0; b < H_B; b++) {
        wr_i[b]         = nt_tape_param_frozen(m->b[b].wr_combined);
        gate_T_i[b]     = nt_tape_param_frozen(m->b[b].gate_T_E);
        gate_inv_T_i[b] = nt_tape_param_frozen(m->b[b].gate_inv_T_E);
        norm1_i[b]      = nt_tape_param_frozen(m->b[b].norm1);
        wq_i[b]         = nt_tape_param_frozen(m->b[b].wq);
        wk_i[b]         = nt_tape_param_frozen(m->b[b].wk);
        wv_i[b]         = nt_tape_param_frozen(m->b[b].wv);
        wo_i[b]         = nt_tape_param_frozen(m->b[b].wo);
        norm2_i[b]      = nt_tape_param_frozen(m->b[b].norm2);
        mg_i[b]         = nt_tape_param_frozen(m->b[b].mlp_gate);
        mu_i[b]         = nt_tape_param_frozen(m->b[b].mlp_up);
        md_i[b]         = nt_tape_param_frozen(m->b[b].mlp_down);
    }
    int norm_f_i = nt_tape_param_frozen(m->norm_f);
    int head_i   = nt_tape_param_frozen(m->out_head);

    /* h0 from tok_emb gather, seeded as NT_OP_NONE — same shortcut as the
     * non-LoRA trainer. tok_emb still frozen, so no missing gradient. */
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
        /* LoRA on wq and wv. wk and wo stay base-only (frozen). */
        int q = nt_lora_forward(wq_i[b], &lora_pairs[b * NUM_TARGETS + TARGET_IDX_WQ], xn, T);
        int k = nt_seq_linear(wk_i[b], xn, T);
        int v = nt_lora_forward(wv_i[b], &lora_pairs[b * NUM_TARGETS + TARGET_IDX_WV], xn, T);
        q = nt_rope_freq(q, T, H_D, 10000.0f);
        k = nt_rope_freq(k, T, H_D, 10000.0f);
        int content = nt_mh_causal_attention(q, k, v, T, H_D);
        int rrpram  = nt_rrpram_lowrank_attention(wr_i[b], xn, v, T, H_E, H_H, H_D);
        int c_scaled = nt_mul(content, gate_T_i[b]);
        int r_scaled = nt_mul(rrpram,  gate_inv_T_i[b]);
        int blend    = nt_add(c_scaled, r_scaled);
        int proj = nt_seq_linear(wo_i[b], blend, T);
        h_i = nt_add(h_i, proj);
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

static float eval_loss(Model *m, nt_lora_pair *lora_pairs, int *toks, long n_toks, int T) {
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
        int loss_idx = forward(m, lora_pairs, in_, tgt_, T);
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
    if (!f) { fprintf(stderr, "[lora] corpus open '%s' failed\n", path); return NULL; }
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

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <base.bin> <corpus.txt> <out_prefix> "
            "[steps=1500] [lr=3e-4] [ctx_T=512] [rank=8] [alpha=16]\n", argv[0]);
        return 2;
    }
    const char *base_path   = argv[1];
    const char *corpus_path = argv[2];
    const char *out_prefix  = argv[3];
    int   steps   = argc > 4 ? atoi(argv[4]) : 1500;
    float base_lr = argc > 5 ? (float)atof(argv[5]) : 3e-4f;
    int   T_ctx   = argc > 6 ? atoi(argv[6]) : 512;
    int   lora_rank  = argc > 7 ? atoi(argv[7]) : 8;
    float lora_alpha = argc > 8 ? (float)atof(argv[8]) : 16.0f;

    fprintf(stderr, "════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Resonance 200M LoRA SFT — Arianna corpus\n");
    fprintf(stderr, "  steps=%d  lr=%.2e  ctx=%d  rank=%d  alpha=%.1f  out=%s\n",
            steps, base_lr, T_ctx, lora_rank, lora_alpha, out_prefix);
    fprintf(stderr, "════════════════════════════════════════════════════════\n");

    Model M;
    memset(&M, 0, sizeof(M));
    int actual_T = read_rs02(base_path, T_ctx, &M);
    if (actual_T < 0) return 1;
    if (actual_T != T_ctx) {
        fprintf(stderr, "[lora] ctx_T capped %d → %d to fit model T_r\n", T_ctx, actual_T);
        T_ctx = actual_T;
    }

    /* Init LoRA pairs: B layers × NUM_TARGETS targets, flat-indexed. */
    int n_pairs = H_B * NUM_TARGETS;
    nt_lora_pair *lora_pairs = calloc((size_t)n_pairs, sizeof(nt_lora_pair));
    for (int i = 0; i < n_pairs; i++) {
        if (nt_lora_init(&lora_pairs[i], H_E, H_E, lora_rank, lora_alpha) != 0) {
            fprintf(stderr, "[lora] nt_lora_init failed at pair %d\n", i);
            return 1;
        }
    }
    long lora_train_p = (long)n_pairs * (lora_rank * H_E + H_E * lora_rank);
    fprintf(stderr, "[lora] adapter: %d pairs × (A %d×%d + B %d×%d) = %.0fK trainable params (%.4f%% of base)\n",
            n_pairs, lora_rank, H_E, H_E, lora_rank, lora_train_p / 1e3,
            100.0 * lora_train_p / (200e6));

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
    float loss_at_1 = 0, loss_at_50 = 0;
    float gnorm_at_1 = 0;
    float val_history[64] = {0};
    int   val_count = 0;
    int   plateau_streak = 0;
    float best_val = 99.0f;
    int   smoke_passed = (steps < 50 ? 1 : 0);

    for (int step = 0; step < steps; step++) {
        float lr = nt_schedule_get_lr(&sched);
        long max_off = n_toks - T_ctx - 1;
        long off = (long)rand() % max_off;
        int *in_  = malloc((size_t)T_ctx * sizeof(int));
        int *tgt_ = malloc((size_t)T_ctx * sizeof(int));
        for (int t = 0; t < T_ctx; t++) { in_[t] = toks[off + t]; tgt_[t] = toks[off + t + 1]; }

        nt_tape_start();
        nt_train_mode(1);
        int loss_idx = forward(&M, lora_pairs, in_, tgt_, T_ctx);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) { first_loss = lv; loss_ema = lv; }
        else loss_ema = 0.95f * loss_ema + 0.05f * lv;

        nt_tape_backward(loss_idx);
        float gn = global_grad_norm();

        if (step == 0)  { loss_at_1 = lv; gnorm_at_1 = gn; }
        if (step == 49 && !smoke_passed) {
            loss_at_50 = lv;
            int killed = 0;
            if (!isfinite(lv)) { fprintf(stderr, "[lora] KILL: NaN at step 50\n"); killed = 1; }
            if (loss_at_50 > loss_at_1) {
                fprintf(stderr, "[lora] KILL: loss[50]=%.4f > loss[1]=%.4f\n", loss_at_50, loss_at_1);
                killed = 1;
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
            float val = eval_loss(&M, lora_pairs, toks, n_toks, T_ctx);
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
                snprintf(best_path, sizeof(best_path), "%s_lora_best.bin", out_prefix);
                if (nt_lora_save(lora_pairs, H_B, NUM_TARGETS, TARGET_NAMES, best_path) != 0) {
                    fprintf(stderr, " [save fail]");
                } else {
                    fprintf(stderr, " ★ best");
                }
            }
            fprintf(stderr, "\n");
            if (val_count < 64) val_history[val_count++] = val;
            if (plateau_streak >= PLATEAU_N) {
                fprintf(stderr, "[lora] EARLY STOP at step %d — val plateaued for %d×%d steps\n",
                        step + 1, PLATEAU_N, VAL_EVERY);
                break;
            }
        }
    }

    /* Final adapter snapshot regardless of best-val. */
    char final_path[512];
    snprintf(final_path, sizeof(final_path), "%s_lora_final.bin", out_prefix);
    nt_lora_save(lora_pairs, H_B, NUM_TARGETS, TARGET_NAMES, final_path);

    double total_s = (now_ms() - t0) / 1000.0;
    fprintf(stderr, "─────────────────────────────────────────────────────\n");
    fprintf(stderr, "[lora] DONE  first %.4f → ema %.4f  (%.0fs / %.1f min)\n",
            first_loss, loss_ema, total_s, total_s / 60.0);
    fprintf(stderr, "[lora] best_val=%.4f  nans=%d  skipped=%d\n",
            best_val, guard.total_nan_count, guard.skipped_steps);
    fprintf(stderr, "[lora] final adapter → %s\n", final_path);

    for (int i = 0; i < n_pairs; i++) nt_lora_free(&lora_pairs[i]);
    free(lora_pairs);
#ifdef USE_CUDA
    if (nt_get_gpu_mode()) gpu_shutdown();
#endif
    return 0;
}
