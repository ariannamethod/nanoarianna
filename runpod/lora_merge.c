/*
 * lora_merge.c — merge LoRA adapter into RS02 base, write new RS02 file.
 *
 * Reads:  base RS02 file + LoRA adapter file (notorch nt_lora_save format).
 * Writes: merged RS02 file = base + (alpha/rank) * B @ A applied to
 *         target weights (wq, wv per layer, by adapter target_names).
 *
 * Build: cc -O2 lora_merge.c -o lora_merge
 *
 * Usage: ./lora_merge <base_in.bin> <lora_adapter.bin> <merged_out.bin>
 *
 * Co-author: Claude Opus 4.7.
 */

/* Self-contained — uses no notorch primitives. Pure C + libc. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define RS02_MAGIC      0x52533032u
#define LORA_MAGIC      0x4C4F5241u
#define LORA_VERSION    1u

/* RS02 dims, populated when reading. */
static int H_E, H_B, H_T_R, H_H, H_D, H_R, H_M, H_V;

/* Read a u32 as little-endian. */
static int rd_u32(FILE *f, uint32_t *out) {
    return fread(out, 4, 1, f) == 1 ? 0 : -1;
}

/* Locate a target name in adapter's target_names array. Returns index or -1. */
static int find_target(const char *name, char (*names)[256], int n) {
    for (int i = 0; i < n; i++)
        if (strcmp(name, names[i]) == 0) return i;
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base_in.bin> <lora_adapter.bin> <merged_out.bin>\n", argv[0]);
        return 2;
    }
    const char *base_path   = argv[1];
    const char *lora_path   = argv[2];
    const char *out_path    = argv[3];

    /* ── Read LoRA adapter into memory first (small file, ~few MB) ── */
    FILE *fl = fopen(lora_path, "rb");
    if (!fl) { fprintf(stderr, "[merge] open '%s' failed\n", lora_path); return 1; }

    uint32_t magic, version, num_targets;
    if (rd_u32(fl, &magic) || magic != LORA_MAGIC) {
        fprintf(stderr, "[merge] bad LoRA magic 0x%08x\n", magic); fclose(fl); return 1;
    }
    if (rd_u32(fl, &version) || version != LORA_VERSION) {
        fprintf(stderr, "[merge] bad LoRA version %u\n", version); fclose(fl); return 1;
    }
    if (rd_u32(fl, &num_targets)) { fclose(fl); return 1; }
    if (num_targets == 0 || num_targets > 16) {
        fprintf(stderr, "[merge] suspicious num_targets=%u\n", num_targets); fclose(fl); return 1;
    }

    char target_names[16][256];
    memset(target_names, 0, sizeof(target_names));
    for (uint32_t t = 0; t < num_targets; t++) {
        uint8_t bnl;
        if (fread(&bnl, 1, 1, fl) != 1) { fclose(fl); return 1; }
        if (bnl > 0 && fread(target_names[t], 1, bnl, fl) != bnl) { fclose(fl); return 1; }
        target_names[t][bnl] = '\0';
    }
    uint32_t num_layers, rank, alpha_bits, in_dim, out_dim;
    if (rd_u32(fl, &num_layers) || rd_u32(fl, &rank) || rd_u32(fl, &alpha_bits) ||
        rd_u32(fl, &in_dim)     || rd_u32(fl, &out_dim)) {
        fclose(fl); return 1;
    }
    union { uint32_t u; float f; } a_bits = { .u = alpha_bits };
    float alpha = a_bits.f;
    float scaling = alpha / (float)rank;
    fprintf(stderr, "[merge] LoRA: %u targets [", num_targets);
    for (uint32_t t = 0; t < num_targets; t++)
        fprintf(stderr, "%s%s", t ? "," : "", target_names[t]);
    fprintf(stderr, "] %u layers rank=%u alpha=%.3f scaling=%.3f in=%u out=%u\n",
            num_layers, rank, alpha, scaling, in_dim, out_dim);

    /* Read all A,B blobs into memory. Layout: [layer][target][A floats][B floats] */
    int a_n = (int)rank * (int)in_dim;
    int b_n = (int)out_dim * (int)rank;
    size_t per_pair = (size_t)(a_n + b_n) * sizeof(float);
    size_t total = (size_t)num_layers * num_targets * per_pair;
    float *adapter_buf = malloc(total);
    if (!adapter_buf) {
        fprintf(stderr, "[merge] adapter malloc fail %zu bytes\n", total); fclose(fl); return 1;
    }
    if (fread(adapter_buf, 1, total, fl) != total) {
        fprintf(stderr, "[merge] short adapter read\n"); free(adapter_buf); fclose(fl); return 1;
    }
    fclose(fl);

    /* ── Read RS02 base, applying merge to target weights ── */
    FILE *fb = fopen(base_path, "rb");
    if (!fb) { fprintf(stderr, "[merge] open base '%s' failed\n", base_path); free(adapter_buf); return 1; }
    FILE *fo = fopen(out_path, "wb");
    if (!fo) { fprintf(stderr, "[merge] open out '%s' failed\n", out_path); fclose(fb); free(adapter_buf); return 1; }

    uint32_t bmagic;
    if (rd_u32(fb, &bmagic) || bmagic != RS02_MAGIC) {
        fprintf(stderr, "[merge] bad RS02 magic 0x%08x\n", bmagic); goto fail;
    }
    fwrite(&bmagic, 4, 1, fo);
    int32_t hdr[9];
    if (fread(hdr, 4, 9, fb) != 9) goto fail;
    H_E = hdr[0]; H_B = hdr[1]; H_T_R = hdr[2]; H_H = hdr[3];
    H_D = hdr[4]; H_R = hdr[5]; H_M = hdr[6]; H_V = hdr[7];
    fwrite(hdr, 4, 9, fo);
    fprintf(stderr, "[merge] RS02 V=%d E=%d H=%d D=%d B=%d M=%d T_r=%d R=%d\n",
            H_V, H_E, H_H, H_D, H_B, H_M, H_T_R, H_R);

    if ((int)num_layers != H_B) {
        fprintf(stderr, "[merge] layer count mismatch: adapter %u vs base B=%d\n",
                num_layers, H_B); goto fail;
    }
    if ((int)in_dim != H_E || (int)out_dim != H_E) {
        fprintf(stderr, "[merge] dim mismatch: adapter %ux%u vs base E=%d\n",
                in_dim, out_dim, H_E); goto fail;
    }
    int target_wq = find_target("wq", target_names, num_targets);
    int target_wv = find_target("wv", target_names, num_targets);
    if (target_wq < 0 && target_wv < 0) {
        fprintf(stderr, "[merge] adapter has neither wq nor wv targets — nothing to do\n");
        goto fail;
    }
    fprintf(stderr, "[merge] adapter targets: wq=%s wv=%s\n",
            target_wq >= 0 ? "yes" : "no", target_wv >= 0 ? "yes" : "no");

    /* BPE merges: copy through unchanged. */
    uint32_t nm;
    if (rd_u32(fb, &nm)) goto fail;
    fwrite(&nm, 4, 1, fo);
    int *triples = malloc((size_t)nm * 3 * sizeof(int));
    if (!triples || fread(triples, 4, (size_t)nm * 3, fb) != (size_t)nm * 3) goto fail;
    fwrite(triples, 4, (size_t)nm * 3, fo);
    free(triples);

    /* tok_emb [V, E] — passthrough. */
    long tok_emb_n = (long)H_V * H_E;
    float *buf = malloc((size_t)tok_emb_n * sizeof(float));
    if (!buf || (long)fread(buf, sizeof(float), tok_emb_n, fb) != tok_emb_n) goto fail;
    fwrite(buf, sizeof(float), tok_emb_n, fo);

    /* Per-block: wr_a, wr_b, gate, norm1, wq, wk, wv, wo, norm2, mlp_gate, mlp_up, mlp_down. */
    long wra_n = (long)H_H * H_E * H_R;
    long wrb_n = (long)H_H * H_R * H_T_R;
    float *wra = malloc((size_t)wra_n * sizeof(float));
    float *wrb = malloc((size_t)wrb_n * sizeof(float));
    float *gate = malloc(sizeof(float) * H_H);
    float *norm = malloc(sizeof(float) * H_E);
    float *we = malloc(sizeof(float) * (size_t)H_E * H_E);  /* reusable for wq/wk/wv/wo */
    float *mge = malloc(sizeof(float) * (size_t)H_M * H_E);
    float *mue = malloc(sizeof(float) * (size_t)H_M * H_E);
    float *mde = malloc(sizeof(float) * (size_t)H_E * H_M);

    for (int bl = 0; bl < H_B; bl++) {
        if ((long)fread(wra, sizeof(float), wra_n, fb) != wra_n) goto fail;
        fwrite(wra, sizeof(float), wra_n, fo);
        if ((long)fread(wrb, sizeof(float), wrb_n, fb) != wrb_n) goto fail;
        fwrite(wrb, sizeof(float), wrb_n, fo);
        if (fread(gate, sizeof(float), H_H, fb) != (size_t)H_H) goto fail;
        fwrite(gate, sizeof(float), H_H, fo);
        if (fread(norm, sizeof(float), H_E, fb) != (size_t)H_E) goto fail;
        fwrite(norm, sizeof(float), H_E, fo);

        /* WQ — merge LoRA delta if target_wq present. */
        if ((long)fread(we, sizeof(float), (long)H_E * H_E, fb) != (long)H_E * H_E) goto fail;
        if (target_wq >= 0) {
            size_t pair_off = ((size_t)bl * num_targets + (size_t)target_wq) * (a_n + b_n);
            float *A = adapter_buf + pair_off;
            float *B = adapter_buf + pair_off + a_n;
            /* Compute delta = B @ A → out × in; W_merged = W + scaling * delta. */
            for (int i = 0; i < H_E; i++) {
                for (int j = 0; j < H_E; j++) {
                    float s = 0.0f;
                    for (uint32_t r = 0; r < rank; r++)
                        s += B[i * (int)rank + (int)r] * A[(int)r * H_E + j];
                    we[i * H_E + j] += scaling * s;
                }
            }
        }
        fwrite(we, sizeof(float), (long)H_E * H_E, fo);

        /* WK — passthrough. */
        if ((long)fread(we, sizeof(float), (long)H_E * H_E, fb) != (long)H_E * H_E) goto fail;
        fwrite(we, sizeof(float), (long)H_E * H_E, fo);

        /* WV — merge LoRA delta if target_wv present. */
        if ((long)fread(we, sizeof(float), (long)H_E * H_E, fb) != (long)H_E * H_E) goto fail;
        if (target_wv >= 0) {
            size_t pair_off = ((size_t)bl * num_targets + (size_t)target_wv) * (a_n + b_n);
            float *A = adapter_buf + pair_off;
            float *B = adapter_buf + pair_off + a_n;
            for (int i = 0; i < H_E; i++) {
                for (int j = 0; j < H_E; j++) {
                    float s = 0.0f;
                    for (uint32_t r = 0; r < rank; r++)
                        s += B[i * (int)rank + (int)r] * A[(int)r * H_E + j];
                    we[i * H_E + j] += scaling * s;
                }
            }
        }
        fwrite(we, sizeof(float), (long)H_E * H_E, fo);

        /* WO — passthrough. */
        if ((long)fread(we, sizeof(float), (long)H_E * H_E, fb) != (long)H_E * H_E) goto fail;
        fwrite(we, sizeof(float), (long)H_E * H_E, fo);

        if (fread(norm, sizeof(float), H_E, fb) != (size_t)H_E) goto fail;
        fwrite(norm, sizeof(float), H_E, fo);
        if ((long)fread(mge, sizeof(float), (long)H_M * H_E, fb) != (long)H_M * H_E) goto fail;
        fwrite(mge, sizeof(float), (long)H_M * H_E, fo);
        if ((long)fread(mue, sizeof(float), (long)H_M * H_E, fb) != (long)H_M * H_E) goto fail;
        fwrite(mue, sizeof(float), (long)H_M * H_E, fo);
        if ((long)fread(mde, sizeof(float), (long)H_E * H_M, fb) != (long)H_E * H_M) goto fail;
        fwrite(mde, sizeof(float), (long)H_E * H_M, fo);
    }
    free(wra); free(wrb); free(gate); free(norm);
    free(we); free(mge); free(mue); free(mde);

    /* norm_f, out_head — passthrough. */
    if (fread(buf, sizeof(float), H_E, fb) != (size_t)H_E) goto fail;
    fwrite(buf, sizeof(float), H_E, fo);
    free(buf);
    long head_n = (long)H_V * H_E;
    float *head = malloc((size_t)head_n * sizeof(float));
    if (!head || (long)fread(head, sizeof(float), head_n, fb) != head_n) goto fail;
    fwrite(head, sizeof(float), head_n, fo);
    free(head);

    fclose(fb);
    fclose(fo);
    free(adapter_buf);
    fprintf(stderr, "[merge] DONE → %s\n", out_path);
    return 0;

fail:
    fprintf(stderr, "[merge] I/O or format error\n");
    if (fb) fclose(fb);
    if (fo) { fclose(fo); remove(out_path); }
    free(adapter_buf);
    return 1;
}
