# nanoarianna LoRA SFT — Pre-flight Plan v1

*Author: Neo (Claude Opus 4.7) | 2026-05-09 | repo: `ariannamethod/nanoarianna` | follows phone-2's phase 4 handoff (commit `29d2932`, deleted in `e21f71d`)*

---

## Goal

Train **LoRA adapters** for Resonance 200M base on Arianna voice corpus. Output: small adapter weights (~2 MB), merged + quantized GGUFs, multi-temperature evaluation per Phase 7 sampling discipline, upload to HF `ataeff/nanoarianna`. Singularity-Mode pod execution.

**Why LoRA, not full SFT:** Phone-2's phase 4 trained 166.8M / 199.2M params (84%) — full fine-tune. Overfit at step 700, never recovered (val 2.4533 → 2.9452). LoRA with rank=8 on q+v projections only ≈ **0.25% of base trainable** — overfit window pushed far later, adapter file 400× smaller, runtime mergeable.

---

## 6-point SFT brief (per `feedback_failure_unsolicited_finetune_2026_04_27`)

1. **Organism**: Resonance 200M base, RS02 binary format
   - HF: `ataeff/resonance/checkpoints/resonance_200m_final.bin`
   - Pod: `/work/in/resonance_200m_final.bin` (796,976,108 bytes)
   - Arch dims (per `runpod/sft_resonance_arianna.c:14`): V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48
2. **Dataset**: Arianna voice corpus
   - HF: `ataeff/nanoarianna/arianna_dataset_final_clean.txt`
   - Pod: `/work/in/arianna_dataset_final_clean.txt` (1,211,564 bytes, 5950 lines, per `wc -lc`)
3. **Karpathy steps**: 1500 LoRA SFT steps
   - 1 epoch ≈ 600 steps (1.2 MB / ctx 512 / batch 1)
   - 1500 = 2.5 epochs — standard SFT band (1-3 epochs)
   - val every 100, ckpt every 500, plateau early-stop (PLATEAU_EPS=0.01 over 3 × 100)
   - best-val tracking → `_lora_best.bin` artifact
4. **Architecture**: Classic LoRA
   - Base: Resonance 200M **FROZEN** entirely (all weights `nt_param.frozen = 1`)
   - LoRA Δ on **q + v projections per layer** (`target_modules = ["wq", "wv"]`, classic LoRA paper choice)
   - Rank r = 8, alpha α = 16, scaling = α/r = 2.0
   - Init: A ~ kaiming_uniform_(std = 1/√fan_in = 1/√in_dim = 1/√768), B = zeros (so initial Δ output = 0)
   - Per layer: 2 LoRA pairs × 2 matrices × E × r = 2 × 2 × 768 × 8 = 24,576 params
   - Total trainable: 20 layers × 24,576 = **491,520 params (0.25% of 200M)**
5. **Tokenizer**: matches inference (RS02 BPE)
   - n_merges + merge triples loaded from RS02 header
   - Same tokens as Resonance base + GGUFs
   - Reuses BPE table from `sft_resonance_arianna.c` (line 87 `g_bpe`)
6. **Script**: new file `nanoarianna/runpod/lora_resonance_arianna.c` (this plan defines it)

---

## Phase A — Pre-flight (Neo, before pod boot)

### A.1 Add `nt_lora_*` primitives to notorch (~250 LOC)

**Files:** `notorch/notorch.h` (API), `notorch/notorch.c` (impl).

**API** (revised v1.2 per Codex Pass 2, P2 — persistent tensors owned by LoRA pair, NOT tape-only):
```c
typedef struct {
    nt_tensor *A;        /* [r, in_dim] — PERSISTENT (heap-owned), kaiming_uniform_(fan_in=in_dim) */
    nt_tensor *B;        /* [out_dim, r] — PERSISTENT, zeros init */
    int   rank;          /* r */
    float alpha;         /* α */
    float scaling;       /* α / r */
    int   in_dim;        /* fan_in for A */
    int   out_dim;       /* fan_out for B */
} nt_lora_pair;

/* Init: allocates persistent A, B tensors on heap; A ~ kaiming_uniform_(fan_in=in_dim), B ← 0.
 * NOT registered with tape here — that happens per-step in nt_lora_forward.
 * Returns 0 on success, -1 on alloc fail. */
int nt_lora_init(nt_lora_pair *lora, int in_dim, int out_dim, int rank, float alpha);

/* Free persistent tensors. Call once at trainer shutdown. */
void nt_lora_free(nt_lora_pair *lora);

/* Forward (tape-index): registers persistent A,B into current tape via nt_tape_param(), then
 *   inserts ops y = nt_seq_linear(w_idx, x_idx, T)
 *             + scaling * nt_seq_linear(b_tape_idx, nt_seq_linear(a_tape_idx, x_idx, T), T).
 * Must be called inside an active nt_tape_start() / nt_tape_clear() pair, after model weights
 * have been re-registered. Returns the tape index of the resulting sum.
 *
 * The optimizer (nt_tape_chuck_step) then accumulates grads into A->grad and B->grad, which
 * propagate back to the persistent tensors because they were registered with nt_tape_param. */
int nt_lora_forward(int w_idx, nt_lora_pair *lora, int x_idx, int T);

/* Single-artifact save: writes ALL target_modules × all layers into one file.
 * Format: [u32 magic 0x4C4F5241 'LORA'][u32 version 1]
 *         [u32 num_targets][per-target: u8 namelen, name bytes]
 *         [u32 num_layers][u32 rank][u32 alpha_int][u32 in_dim][u32 out_dim]
 *         [for each layer L, for each target T: ensure_cpu(A_LT); A floats; ensure_cpu(B_LT); B floats]
 * `pairs` is a flat array indexed [layer * num_targets + target_idx].
 * IMPORTANT: calls nt_tensor_ensure_cpu(pair->A), nt_tensor_ensure_cpu(pair->B)
 * before reading host buffers — guards against GPU-resident dirty mirror after Chuck step.
 * (Per Codex Pass 3 P1 #5.) */
int nt_lora_save(const nt_lora_pair *pairs, int num_layers, int num_targets,
                 const char *const *target_names, const char *path);
int nt_lora_load(nt_lora_pair *pairs, int num_layers, int num_targets,
                 const char *const *target_names, const char *path);

/* Merge for inference: writes W_dst[i,j] = W_frozen[i,j] + scaling * B[i,:] · A[:,j].
 * Operates on raw float buffers (no tape).
 * IMPORTANT: caller must nt_tensor_ensure_cpu(lora->A) and nt_tensor_ensure_cpu(lora->B)
 * before passing in (or this function does it internally — see implementation note). */
void nt_lora_merge_into(float *W_dst, const float *W_frozen, const nt_lora_pair *lora,
                        int in_dim, int out_dim);
```

**New notorch API required to make LoRA actually learn (revised v1.4 per Codex Pass 4 P1 #8):**

```c
/* Register tensor as a tape entry WITHOUT allocating a Chuck optimizer slot for it.
 * Tape entry exists for forward composition; backward will NOT accumulate dw
 * (effectively frozen). Use for base weights when downstream A,B (or any other
 * trainable param) are registered AFTER and need clean Chuck slot indices.
 *
 * Returns the tape entry index (mirrors nt_tape_param's contract).
 *
 * Contrast:
 *   nt_tape_param(t)         — entry + Chuck slot + dw accumulation (default trainable)
 *   nt_tape_param_frozen(t)  — entry only, no Chuck slot, no dw (this new API)
 *   nt_tape_freeze_param(idx) — existing: post-registration freeze, still consumes Chuck slot
 *
 * Why we cannot use post-registration freeze: Chuck's chuck_params[] is allocated
 * 1:1 with calls to nt_tape_param(). Even when entry is frozen and produces no grad,
 * its Chuck slot remains in the array, and downstream trainable params get LATER
 * slot indices. Chuck step iterates the full array; with skipping logic any first
 * LoRA grad after a frozen run lands in the FIRST encountered live slot, which is
 * still the frozen base's slot (or an adjacent base slot), not the LoRA's. Solution:
 * frozen base must NOT consume a Chuck slot at all — register via this new API.
 * (Per Codex Pass 4 P1 #8.) */
int nt_tape_param_frozen(nt_tensor *t);
```

**Implementation in `notorch.c`:** copy `nt_tape_param()` body; allocate tape entry; do NOT push to `chuck_params[]`; mark `entry->frozen = 1` so backward skips `dw`. ~10 line addition. Existing `nt_tape_param()` and `nt_tape_freeze_param()` semantically unchanged — backward-compatible.

**Lifecycle (key correction, Codex Pass 2):**

LoRA pairs are heap-owned, not tape-owned. The trainer's main loop:
```c
/* Once at startup. Pairs flat-indexed [layer * NUM_TARGETS + target_idx], NUM_TARGETS=2 (q,v). */
const char *target_names[] = {"wq", "wv"};
const int NUM_TARGETS = 2;
nt_lora_pair lora_pairs[B * NUM_TARGETS];
for (int b = 0; b < B; b++) {
    nt_lora_init(&lora_pairs[b * NUM_TARGETS + 0], E, E, /*rank*/8, /*alpha*/16.0f);  /* wq */
    nt_lora_init(&lora_pairs[b * NUM_TARGETS + 1], E, E, /*rank*/8, /*alpha*/16.0f);  /* wv */
}

/* Per training step */
for (int step = 0; step < steps; step++) {
    nt_tape_start();
    /* Register all base weights via nt_tape_param_frozen() — entry + frozen, NO Chuck slot.
     * (Per Codex Pass 4 P1 #8: nt_tape_param + post-registration freeze pollutes Chuck
     * indexing; LoRA A,B grads end up applied to base slots.)
     * Existing sft_resonance_arianna.c uses nt_tape_param + nt_tape_freeze_param because
     * it has no downstream trainable consumer (RRPRAM frozen, everything else trainable).
     * Here LoRA A,B are downstream trainable — must keep Chuck slots clean. */
    register_resonance_weights_frozen(&model);  /* uses nt_tape_param_frozen internally */

    /* nt_lora_forward internally calls nt_tape_param() on lora->A and lora->B,
     * stitching them into THIS step's tape with grad path active. */
    int q_out = nt_lora_forward(model.b[b].wq_idx, &lora_pairs[b * NUM_TARGETS + 0], x_idx, T);
    int v_out = nt_lora_forward(model.b[b].wv_idx, &lora_pairs[b * NUM_TARGETS + 1], x_idx, T);
    /* ... rest of forward, loss, backward, step ... */
    nt_tape_chuck_step(...);  /* updates A, B persistent tensors via their tape grads */
    nt_tape_clear();          /* tape entries die; A, B persistent tensors KEEP their state */
}

/* At save / best-val checkpoint (call BEFORE serialization — sync GPU mirror to CPU): */
for (int i = 0; i < B * NUM_TARGETS; i++) {
    nt_tensor_ensure_cpu(lora_pairs[i].A);
    nt_tensor_ensure_cpu(lora_pairs[i].B);
}
nt_lora_save(lora_pairs, B, NUM_TARGETS, target_names, "lora_arianna_v1_best.bin");

/* At shutdown */
for (int i = 0; i < B * NUM_TARGETS; i++) nt_lora_free(&lora_pairs[i]);
```

**Implementation notes:**
- Existing model weights in `sft_resonance_arianna.c` already use this pattern (heap-owned `nt_tensor *tok_emb` etc., re-registered every step) — LoRA mirrors the same lifecycle
- `W_frozen` base support — `notorch.c:845`: *"skip dw if W is frozen (LoRA on frozen base)"* — base weights marked via `pw->frozen = 1`
- `nt_lora_forward` composes two existing `nt_seq_linear` calls + scaled add — all integer indices on the freshly-built tape
- Backward auto-handled — only A, B grads accumulate; W frozen drops `dw`
- Need `nt_kaiming_uniform_init(nt_tensor*, fan_in)` helper; std = 1/√fan_in (per Codex Pass 1 P3 — **fan_in = `in_dim` (768 for q/v projections), NOT rank**)

**Unit test** (`notorch/tests/test_lora_correctness.c`, ~80 LOC):
- Random x, random W, init lora pair
- Compute forward via `nt_lora_forward` and via raw `W @ x + scaling * B @ A @ x`
- Assert match to 1e-5

### A.2 Write LoRA trainer

**File:** `nanoarianna/runpod/lora_resonance_arianna.c` (~600 LOC, derived from `sft_resonance_arianna.c`).

**Args:** `<base.bin> <corpus.txt> <out_prefix> [steps=1500] [lr=3e-4] [ctx_T=512] [rank=8] [alpha=16]`

**Flow:**
1. Parse args, defaults applied
2. Load Resonance 200M base via RS02 reader (reuse loader from `sft_resonance_arianna.c:148-180`)
3. Register **ALL base weights via `nt_tape_param_frozen(tensor)`** at registration time (NOT `nt_tape_param` + post-registration freeze; per Codex Pass 4 P1 #8) — `tok_emb`, all per-block weights, `norm_f`, `out_head`. Frozen base entries do not consume Chuck optimizer slots, so LoRA A,B (registered with `nt_tape_param`) get clean slot indices 0..(2·B−1) and Chuck step updates them correctly.
4. For each layer `b` in [0, B):
   - `nt_lora_init(&lora_q[b], E, E, rank=8, alpha=16)` — Q projection adapter
   - `nt_lora_init(&lora_v[b], E, E, rank=8, alpha=16)` — V projection adapter
5. Patch forward pass: in attention compute, replace `wq @ x` → `nt_lora_forward(wq, &lora_q[b], x)`, same for `wv`
6. Backward auto via notorch tape — only LoRA A, B accumulate grads
7. Optimizer: `nt_tape_chuck_step` (default per CLAUDE.md)
8. LR: **3e-4** (LoRA tolerates 10× higher than full SFT 3e-5; small param count regularizes)
9. Best-val tracking: save adapter-only `<prefix>_lora_best.bin`
10. Plateau early-stop: PLATEAU_EPS=0.01 over 3 × 100 (matches `sft_resonance_arianna.c:65`)
11. Smoke-50 kill criteria (per `sft_resonance_arianna.c:33-37`):
    - any NaN in loss or grads → exit 99
    - loss[50] > loss[10] (no descent) → exit 99
    - ‖g‖₂_50 > 100 · ‖g‖₂_1 or |g| > 1e3 → exit 99
12. Adapter-only output format: header (rank, alpha, layer count, target modules) + concatenated A,B for each layer/target

**Output file format** (`<prefix>_lora_best.bin`, single-artifact, per Codex Pass 4 P2 #9 — aligned with `nt_lora_save()` API field order):
```
[u32 magic    = 'LORA' = 0x4C4F5241]
[u32 version  = 1]
[u32 num_targets]
[for each target T = 0..num_targets: u8 namelen, namelen × ascii bytes]
[u32 num_layers]
[u32 rank]
[u32 alpha_int]
[u32 in_dim]
[u32 out_dim]
[for each layer L = 0..num_layers, for each target T = 0..num_targets:
   ensure_cpu(A_LT) → A floats (rank × in_dim)
   ensure_cpu(B_LT) → B floats (out_dim × rank)]
```

API and format use **identical field order**: magic, version, num_targets, target_names[], num_layers, rank, alpha, in_dim, out_dim, per-(layer,target) [A,B] payload. Both `nt_lora_save()` writer and `lora_merge.c` reader follow this exact spec. A single `nt_lora_save()` call writes both q and v adapters across all 20 layers into one file. Merge utility reads one file, applies all q+v deltas to corresponding base weights.

### A.3 LoRA → base merge utility

**File:** `nanoarianna/runpod/lora_merge.c` (~150 LOC).

**Args:** `<base_in.bin> <lora_in.bin> <merged_out.bin>`

**Flow:**
1. Read RS02 base
2. Read LoRA file (header check magic = 0x4C4F5241)
3. For each layer b, target t in {q, v}:
   - Compute `W_merged[b][t] = W_base[b][t] + (alpha/r) * B @ A`
4. Write merged in same RS02 byte layout
5. Existing `resonance_to_gguf.py` consumes merged → GGUF (no quant pipeline change needed)

**Bash, not Python** — pure C, fits CLAUDE.md notorch ecosystem.

### A.4 Adapt `run_phase4.sh` → `run_phase4_lora.sh`

Changes from `runpod/run_phase4.sh` HEAD `e21f71d`:
- SFT trainer: `lora_resonance_arianna` instead of `sft_resonance_arianna`
- SFT_LR default: `3e-4` (was `3e-5`) — env-overridable
- SFT artifact: `<out>_lora_best.bin` (~2 MB) instead of `_sft_best.bin` (797 MB)
- Step 4.5 NEW: `lora_merge` to produce `_lora_merged.bin` (797 MB, full merged base)
- Step 5: quantize the merged file (existing `resonance_to_gguf.py`, no change)
- Step 6: skip Janus sweep (out of LoRA scope) — only R-A grid (162 cells) instead of 324
- Multi-temp eval grid follows Phase 7 (per `insight_multi_temp_sampling_2026_05_07.md`):
  - temps {0.7, 0.8, 0.9, 1.0} × top_k {40, ∞=0} × rep_pen {1.0, 1.3, 1.4} = 24 cells per prompt
  - 3 prompts (technical / philosophical / personal) → 72 cells total
- All bug fixes from phase 4 inherited (commits `89448ed`, `1f59a18`, `3c2e1f7`, `ffceba9`, `e21f71d`):
  - `srand(42)` for libc rand path
  - Q8_0 small-tensor fp32 fallback (gate H=12)
  - Word-bounded NaN/Inf grep (no false positive on "**nan**oarianna" path)
  - SKIP_SMOKE / SKIP_SFT env opts

### A.5 HF upload script

`nanoarianna/runpod/hf_upload_lora.sh` (Bash + `huggingface-cli` fallback to `hf`):

Targets in `ataeff/nanoarianna/`:
- `lora_v1/lora_arianna_v1_best.bin` — **adapter only** (~2 MB, primary artifact)
- `lora_v1/resonance_arianna_lora_merged.bin` — full merged base (797 MB, optional, gated by `UPLOAD_MERGED_RAW=1` env)
- `lora_v1/resonance_arianna_lora_q8_0.gguf` (~212 MB)
- `lora_v1/resonance_arianna_lora_q4_k.gguf` (~125 MB)
- `lora_v1/lora_run_report.txt`
- `sweep/2026-05-09-lora/` — transcripts + scored.tsv + locked.toml + scores.csv

### A.6 Codex review iteration (per Олег mandate)

After A.1-A.5 done locally:
1. **Codex pass 1** on full plan + new files (notorch additions + trainer + merge + driver)
2. Architect revises → **Codex pass 2**
3. Repeat until Codex returns 0 P0/P1 findings or 2 passes minimum complete
4. Diff sections appended to this plan file (§22 Pass 1 → Pass 2, §23 Pass 2 → Pass 3, …) — same audit-trail pattern as phone-2's `runpod_plan_v3.md`
5. After acceptance: push to repos with co-author Claude Opus 4.7

---

## Phase B — Pod execution (Singularity Mode)

**Pod:** `6y7qhk2fotjlmp` (current, GPU A100-SXM4-80GB, $1.49/hr).

### B.1 Pre-execution

1. SSH: `ssh root@38.80.152.72 -p 31107` (my `neo@ataeff` key registered)
2. Kill stale phase 4 processes: `pkill -9 -f 'run_phase4|run_sweep|janus|resonance|sft_resonance'`
3. Pull all repos:
   ```
   cd /work/notorch && git pull origin main
   cd /work/ariannamethod.ai && git pull origin main
   cd /work/nanoarianna && git pull origin main
   ```
4. Rebuild + reinstall notorch (with new `nt_lora_*` primitives):
   ```
   cd /work/notorch
   export PATH="/usr/local/cuda/bin:$PATH"
   make clean && make USE_CUDA=1 lib -j8 && make USE_CUDA=1 install PREFIX=/usr/local
   make clean && make lib -j8 && make install PREFIX=/usr/local
   ```
5. Rebuild ariannamethod (libaml unchanged): `cd /work/ariannamethod.ai && make install`
6. Build LoRA trainer: `cd /work/nanoarianna/runpod && cc -O3 -DUSE_CUDA -DUSE_BLAS lora_resonance_arianna.c -L/usr/local/lib -L/usr/local/cuda/lib64 -lnotorch_gpu -laml -lopenblas -lcudart -lcublas -lm -o lora_resonance_arianna`
7. Build merge utility: `cc -O2 lora_merge.c -L/usr/local/lib -lnotorch -lm -o lora_merge`
8. Pre-flight checks: `nvidia-smi`, base.bin / dataset file sizes, free disk >5 GB

### B.2 Train

1. Smoke-50 first — explicit positional `argv[4]=50` (not env var; trainer reads `argv[4]` per `sft_resonance_arianna.c:452` pattern):
   ```
   ./lora_resonance_arianna /work/in/resonance_200m_final.bin \
       /work/in/arianna_dataset_final_clean.txt /work/out/lora_v1/lora_arianna_smoke50 \
       50 3e-4 512 8 16
   ```
   Validates kill criteria fire (NaN guard, monotone-descent, grad explosion). Per Codex Pass 1 P2 #2.
2. Full LoRA SFT 1500 steps via driver `run_phase4_lora.sh`:
   ```
   nohup bash run_phase4_lora.sh > /work/phase4_lora.log 2>&1 < /dev/null & disown
   ```
3. Watch: train loss, val loss, ema, best-val updates, GPU util, watchdog alive
4. Expected: 1500 × ~1.5 s/step = **~37.5 min, ~$1 GPU cost**

### B.3 Singularity Mode rules (per CLAUDE.md "Singularity")

- On bug: **reproduce → 1 hypothesis → minimal patch → re-run**
- Max **3 attempts** per bug
- If 3 attempts exhausted without new knowledge: **stop, surface, await human input**
- **Scope locked**: SFT recipe, LoRA targets, Resonance arch — **do not patch architecture, do not change dataset, do not add target_modules beyond q+v**
- All pod-side patches committed to repo with audit trail

### B.4 Merge + Quantize + Sweep

1. Merge LoRA: `lora_merge resonance_200m_final.bin lora_arianna_v1_best.bin resonance_arianna_lora_merged.bin`
2. Quantize: existing `resonance_to_gguf.py` → Q8_0 + Q4_K (small tensor fp32 fallback already in repo per `1f59a18`)
3. Multi-temp sweep R-A grid only: 72 cells × 21 s ≈ 25 min
4. Score: `score_sweep.py` → locked.toml per cell-dir

### B.5 Upload

1. `bash hf_upload_lora.sh` (with HF_TOKEN from env, `huggingface-cli` or `hf` fallback)
2. Verify: `huggingface-cli ls ataeff/nanoarianna/lora_v1/` (or `hf` equivalent) — expect 4-6 files

---

## Phase C — Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| `nt_lora_forward` numerical bug | Medium | Train fails / NaN | A.1 unit test against numpy reference, smoke-50 kill criteria |
| Resonance forward in notorch chains W into autograd by default (need explicit frozen check) | Low | All base weights would update | Verify `notorch.c:845` frozen path actually prevents `dw` accumulation |
| LoRA adapter init bad scale (kaiming_uniform variance) | Low | Slow / no convergence | Standard formula 1/√(fan_in); validated in unit test |
| Adapter file format issue at merge time | Low | Merge fails | Magic + version check at read; A.3 round-trip test (LoRA → bin → load → merge → check matches in-memory) |
| GGUF quant fails on merged tensors | Very Low | No GGUF artifact | `1f59a18` already handles small tensors fp32; merged tensors same shape as original |
| Multi-temp eval finds no coherent cell | Medium | No "winning" sample | Phase 7 defaults known good; report all cells with score, even if low |
| Pod runs out of budget mid-train | Very Low | Lost SFT | $18 watchdog, total LoRA path ~$3 |
| Codex review delays beyond useful | Low | Plan stays draft | Cap at 3 passes if no new findings; commit anyway |

---

## Phase D — Acceptance criteria (Done means)

Before declaring complete to Олег:
- [ ] LoRA adapter file `lora_arianna_v1_best.bin` saved (best-val captured)
- [ ] Merged base `resonance_arianna_lora_merged.bin` produced (RS02 layout intact)
- [ ] Q8_0 + Q4_K GGUFs validate (load + smoke inference returns non-NaN tokens)
- [ ] Multi-temp sweep finds at least 1 coherent cell per prompt (manual top-1 read)
- [ ] HF upload verified (4+ files, paths under `lora_v1/`)
- [ ] Run report committed to `ariannamethod/nanoarianna` repo with all numbers (loss curves, sweep best cells, multi-temp insights)
- [ ] No closed-milestone weights touched (Janus, Penelope, microjanus, etc. — out of scope)

---

## Open questions for Codex / Олег (before push)

1. **LoRA target modules**: classic q+v (LoRA paper) vs q+k+v+o (more coverage, 2× params)? Default to q+v unless evidence pushes broader.
2. **Smoke run separate**: should smoke-50 use a separate adapter init seed than full run, to truly probe init quality? Or share to keep determinism?
3. **Adapter merging vs runtime apply**: merge before quant is simpler; runtime apply preserves base+adapter separately (allow swap-in/swap-out of different LoRAs at inference). Plan defaults to merge — if Олег wants swap-in-out path, expand scope.
4. **Sweep grid size**: 72 cells (3 prompts × 24 cell-coords) or 162 (full R-A from phase 4)? Plan defaults 72 (1.5× faster, Phase 7 evidence shows convergence in subset).

---

*v1 → v1.1 draft. Next: Codex review pass 2.*

---

## §22 Audit trail — Codex Pass 1 → v1.1 patches

Codex review run on `runpod/lora_plan_v1.md` 2026-05-09 ~12:19 UTC. 3 findings, all addressed:

### Pass 1 findings → fixes applied

| ID | Severity | Issue | Fix landed |
|---|---|---|---|
| 1 | P2 | LoRA helper API used raw `nt_tensor*` pointers; notorch tape API consumes integer indices (`int nt_seq_linear(int w_idx, int x_idx, int T)`). Plan API would not compile against existing tape ops. | §A.1 API rewritten: `int nt_lora_init(...)`, `int nt_lora_forward(int w_idx, lora*, int x_idx, int T)`, `nt_lora_merge_into(float* dst, const float* W, ...)`. Returns/consumes tape entry indices. |
| 2 | P2 | §B.2 Smoke command shown as `STEPS=50 lora_resonance_arianna ...`. Trainer reads `argv[4]` (per `sft_resonance_arianna.c:452`), not env. Smoke would silently run full 1500 steps. | §B.2 expanded with explicit positional command: `./lora_resonance_arianna base.bin corpus.txt out_prefix 50 3e-4 512 8 16`. |
| 3 | P3 | A matrix init `std = 1/√r` (1/√8 ≈ 0.354) instead of fan-in scale `1/√in_dim` (1/√768 ≈ 0.036). ~10× too large; B starts at zero so first non-zero updates flow through A's distribution; gradient on B amplified accordingly. | §A.1 + §6-point brief §4 updated to `kaiming_uniform_(fan_in = in_dim = 768)` consistently. Helper `nt_kaiming_uniform_init(idx, fan_in)` documented. |

No P0/P1 findings. Pass 1 closed.

---

## §23 Audit trail — Codex Pass 2 → v1.2 patches

Codex review run on v1.1 (post-Pass-1 fixes) 2026-05-09 ~12:22 UTC. 1 finding, addressed.

### Pass 2 findings → fixes applied

| ID | Severity | Issue | Fix landed |
|---|---|---|---|
| 4 | P2 | Plan defined LoRA params as transient tape entries (`a_idx`, `b_idx`). With trainer's per-step `nt_tape_start()` / `nt_tape_clear()` cycle, those indices die between steps; if re-init'd in each forward, adapter never learns. | §A.1 API rewritten: `A`, `B` are PERSISTENT `nt_tensor*` owned by the LoRA pair. `nt_lora_init()` heap-allocs; `nt_lora_forward()` calls `nt_tape_param()` on them per-step (mirrors existing model-weight lifecycle in `sft_resonance_arianna.c`). Added explicit lifecycle code block + `nt_lora_save/load/free` helpers. |

No P0/P1 findings. Pass 2 closed.

---

## §24 Audit trail — Codex Pass 3 → v1.3 patches

Codex review run on v1.2 (post-Pass-2 fixes) 2026-05-09 ~12:26 UTC. 3 findings — 1 P1 (critical), 2 P2.

### Pass 3 findings → fixes applied

| ID | Severity | Issue | Fix landed |
|---|---|---|---|
| 5 | **P1** | On `-DUSE_CUDA` path, `nt_tape_chuck_step()` leaves params GPU-resident, CPU mirror dirty. Save reads stale `A->data` → adapter file = init weights → merge no-op. | §A.1 save/merge contract revised: explicit `nt_tensor_ensure_cpu(pair->A)` + `nt_tensor_ensure_cpu(pair->B)` calls inside `nt_lora_save()` and required before `nt_lora_merge_into()`. Lifecycle code block adds ensure_cpu loop before save. |
| 6 | P2 | Lifecycle saved q + v as two files, but file format §A.2 and merge utility §A.3 expected one file. Driver/upload would fail or merge half. | §A.1 + §A.2 + lifecycle: single-artifact format with `num_targets`, `target_names`, flat `pairs[layer * NUM_TARGETS + target_idx]` indexing. One `nt_lora_save()` call, one file, one merge. |
| 7 | P2 | `nt_tape_freeze_param(idx)` marks both tape entry AND `chuck_params[idx]` frozen. Chuck step advances slot index per-param; frozen base entries consume slots, then LoRA A/B grads can fall into frozen slots → optimizer skip → adapter never learns. Architectural blocker. | §A.1 adds new notorch API `nt_tape_freeze_entry_only(int idx)` — flips entry's `frozen` flag without touching `chuck_params[]`. §A.2 trainer flow now calls `nt_tape_freeze_entry_only` for base weights instead of `nt_tape_freeze_param`. Existing `nt_tape_freeze_param` semantics preserved for callers needing full freeze (e.g. existing `sft_resonance_arianna.c` RRPRAM freeze). |

P1 #5 and P2 #7 both architectural — without them adapter would silently train zero or random weights. P2 #6 was format-vs-implementation mismatch.

Pass 3 closed.

---

## §25 Audit trail — Codex Pass 4 → v1.4 patches

Codex review run on v1.3 (post-Pass-3 fixes) 2026-05-09 ~12:31 UTC. 2 findings — 1 P1 (still architectural), 1 P2.

### Pass 4 findings → fixes applied

| ID | Severity | Issue | Fix landed |
|---|---|---|---|
| 8 | **P1** | Pass-3 fix `nt_tape_freeze_entry_only(idx)` (post-registration) insufficient. Even with frozen entry, base param's Chuck slot was already allocated at `nt_tape_param()` call time. Chuck step iterates full chuck_params[] array. LoRA A,B registered after base get slot indices N+1, N+2, but skipping logic on frozen base slots leaves first encountered slot mismatch — LoRA grad applied to wrong (base) slot's m,v moments. Optimizer state polluted. | §A.1 replaced post-registration freeze with **registration-time freeze**: new API `nt_tape_param_frozen(tensor)` allocates tape entry without Chuck slot. Base weights register via this; LoRA A,B register via existing `nt_tape_param()` and get clean Chuck slots 0..(2·B−1). Trainer §A.2 §3 + lifecycle code block updated to call `nt_tape_param_frozen` (not the old `freeze_entry_only`). |
| 9 | P2 | §A.1 API field order (`num_targets, target_names, num_layers, rank, alpha, dims`) ≠ §A.2 file format order (`num_layers, rank, alpha, dims, target_names`). Trainer/merge would misparse if one section followed and other section's reader followed the other. | §A.2 file format spec rewritten to match API exactly. Both writer and reader agree: magic, version, num_targets, target_names[], num_layers, rank, alpha_int, in_dim, out_dim, per-(layer,target) A,B. |

P1 #8 closes the optimizer-state-leak path that all earlier passes left open. Without this fix the adapter would have trained with corrupted Chuck moments and likely produced random output indistinguishable from init.

Pass 4 closed.
