# Phase 4 — RunPod brief (v3, post-second-audit)

**Goal:** produce two working voice cells for phone-2 — **Slot A = Janus 176M + Leo SFT** and **Slot B = Resonance 200M + Arianna SFT (NEW)** — locked sampling parameters, GGUF-quantized, uploaded to `huggingface.co/ataeff/nanoarianna`, and verified in dialogue cycle on phone-2.

**v3 changes from v2 (after Opus audit pass 2):**

- **A.1+A.2 RESOLVED in code, not just spec.** Both organism wrappers patched in this same commit: `organism/janus.aml` exposes `g_yent_top_k` and `g_yent_rep_pen` globals + new `--top-k` / `--rep-pen` argv flags; `organism/tools/resonance_forward.h` (now vendored locally — 16 KB, controls our sampler) exposes `g_resonance_rep_pen` + `--rep-pen` flag in `organism/resonance.aml`. Build verified: `make all` produces both binaries clean (Janus 536 KB, Resonance 250 KB), flags accepted at runtime.
- **B.1 RESOLVED.** Resonance dimensions corrected against actual `huggingface.co/ataeff/resonance/model.py:222-231` `RESONANCE_200M`: `n_embd=768, n_head=12, head_dim=64, n_layer=20, rrpram_rank=48, context_len=2048, ffn_dim=2048, vocab_size=16384`. v2's table had every number wrong (audit B.1).
- **B.2 RESOLVED.** RRPRAM low-rank training-mode through notorch tape **is supported and grad-check verified.** `nt_rrpram_lowrank_attention` defined at `notorch.c:3232`; backward path persisted via tape (`notorch.c:3258-3259` "rrpram U buffer ... persisted to backward via tape"); CPU + GPU paths exist (`notorch.c:1271-1310`); finite-diff grad check at `tests/test_rrpram_lr.c` PASSES on aarch64-Termux phone-2 (`max_rel_diff 6.06e-02, fails 0` — measured during phone-2 Phase 1, see `nanoarianna/ECOSYSTEM_LOG.md`). v2 audit B.2 over-stated this as "unverified multi-day lift"; the lift is real but bounded.
- **D.1 RESOLVED.** RS02 format spec added inline (see §"Phase 4-α SFT" below).
- **B.3 fixed.** Q5_K_M references replaced by Q5_0 / Q6_K (only formats `notorch/gguf.c:354` actually decodes).
- **B.4-B.6, D.2, F.1, F.2 addressed** in their respective sections.
- **H.1 budget realism:** $10–18 retained, with explicit RRPRAM-training contingency clause (see §"Open decisions" §1).

**Outputs of this phase:**

1. `huggingface.co/ataeff/nanoarianna` populated with:
   - `slot_arianna/` — Resonance 200M Arianna SFT GGUF Q8_0 + Q4_K + locked `init_arianna.aml`
   - `slot_leo/` — Janus 176M Leo SFT GGUF Q8_0 + Q4_K + locked `init_leo.aml`
   - `sft_v2/resonance_200m_sft_arianna.bin` — raw fp32 RS02 SFT output (797 MB), **persisted before sweep starts** (audit D.2 — protects against watchdog kill mid-sweep)
   - `sweep/<date>/` — full sweep archive (TSVs + transcripts + locked.toml per cell)
   - `manifest.toml` (schema below — audit F.2)
   - `LICENSE-WEIGHTS` (Janus Identity License v1.0; source `~/resonance.aml/LICENSE-WEIGHTS` 1439 bytes — verified)
2. `nanoarianna/personas/init_arianna.aml` and `nanoarianna/personas/init_leo.aml` updated with measured per-voice optimal `(temp, top_k|top_p, rep_pen)` baked in.
3. Run report at `~/ariannamethod/phones/results/galaxy-a07/<date>-runpod-sweep.md`. **Mac Neo Architect handoff** (audit F.1): the run-report file IS the handoff — Mac Neo pulls via `git pull` of umbrella repo and writes a review section at the bottom (same shape as `2026-05-07-10k-char-arianna-final.md` Architect review). Cross-link from `nanoarianna/ECOSYSTEM_LOG.md`.

---

## Context

(Identical to v2 brief context — preserved verbatim.) Phone-2 closed Phase 3 clean (`https://github.com/ariannamethod/nanoarianna` HEAD post-this-commit; pre-this commit `ea2b8f9`). Phase 4 produces real weights and per-voice optimal sampling.

Dario paper §5.7 / Result 7 / Appendix C: *"Sampling is not a decoding parameter. Sampling is a state-space entry condition. A checkpoint is not dead until it has been swept."* Each cell gets its full grid sweep at the architecture's natural sampler.

---

## 6-point training brief (Phase 4-α SFT, Oleg-approved 2026-05-09)

Per `memory/feedback_failure_unsolicited_finetune_2026_04_27.md`. All six concrete, no defaults.

1. **Organism:** Resonance 200M base. Per `huggingface.co/ataeff/resonance/model.py:222-231` `RESONANCE_200M`:

   | knob | value |
   |---|---|
   | n_embd     | 768 |
   | n_head     | 12 |
   | head_dim   | 64 |
   | n_layer    | 20 |
   | rrpram_rank | 48 |
   | context_len | 2048 |
   | ffn_dim    | 2048 |
   | vocab_size | 16384 |

   Architecture: dual attention (Content + RRPRAM low-rank, R=48), parametric RMSNorm, sigmoid per-head gate, even/odd RoPE θ=10000, SwiGLU MLP. Defined in `~/resonance.aml/tools/resonance_forward.h` (state_dict layout at `:94-114`). **Authoritative source for runtime dims is the RS02 header** (read at `resonance_forward.h:265-268`); the SFT trainer reads the header rather than encoding compile-time constants. Base weights at `huggingface.co/ataeff/resonance/checkpoints/resonance_200m_final.bin` (797 MB raw fp32, RS02 magic) — verified byte-exact via HF API HEAD this session.

2. **Dataset:** `huggingface.co/ataeff/nanoarianna/arianna_dataset_final_clean.txt` — 1,211,564 bytes (1.21 MB) hand-curated Arianna corpus. Verified byte-exact via HF API HEAD; same bytes as the corpus that produced bit-identical char-level loss 5.5804 → 1.0685 on phone-1 and phone-2 (`memory/milestone_phone2_galaxy_a07_10k_2026_05_07.md`). Conforms to `feedback_no_default_datasets_2026_04_27.md` (hand-curated, not upstream default). For SFT-format probing also available: `arianna_en_sft.jsonl` (2,401,747 bytes / 2.40 MB, Q/A pairs).

3. **Karpathy steps:** SFT regime, **1500 steps, val every 100, ckpt every 500.** Effective epoch count to be reported by trainer at startup as `corpus: X.X MB → N tokens (epochs/step = N_tokens / (batch_size × ctx))`. Default `batch_size=4, ctx=2048` → 8192 tokens per step. Arianna corpus tokenized through Resonance's existing 16128-merge BPE: ClimbMix-trained BPE on Arianna text typically yields ~0.25 BPE-tokens/byte, so 1.21 MB × 0.25 ≈ **300K tokens**, giving ~**37 steps per epoch** at batch_size=4 ctx=2048; 1500 steps ≈ **40 epochs** (audit B.4 — explicit estimate now). **Pre-flight tokenizer-compression check:** before SFT starts, run `./resonance_bpe_check arianna_dataset_final_clean.txt` to print actual `tokens/byte` ratio; if < 0.20, halt and report (signal that ClimbMix BPE is bloating Arianna text — Phase 5 decision whether to re-merge BPE on Arianna corpus instead) (audit B.6).

   **Early-stop, tightened (audit B.5):** evaluate val loss **every 100 steps**, not every 500. Plateau definition: `|val[t] − val[t−100]| < 0.01` for **3 consecutive 100-step val checks** (300 steps to detect, not 1500). If `val[t] > val[t-100]` (rising), stop immediately at the rising checkpoint. Final ckpt = best-val ckpt observed.

4. **Architecture:** Resonance dual-attention (per item 1 above). **NOT char-level** — Resonance is BPE-token-only at vocab_size=16384 per `RESONANCE_200M`. Confirms `feedback_no_char_models.md` (BPE preferred; char-level only with explicit reason).

5. **Tokenizer:** Resonance's existing 16128-merge BPE from `huggingface.co/ataeff/resonance/checkpoints/tokenizer.bin` (193,540 bytes — verified byte-exact via HF API HEAD). **Same tokenizer at inference** — Slot B inference reads merges from RS02 file's embedded merge table at load time (`resonance_forward.h:272-281`). Tokenizer SHA captured at SFT start (`sha256sum tokenizer.bin > runs/sft_arianna/tokenizer.sha256`) and re-verified post-conversion.

6. **Script:** `notorch + Chuck` SFT path. Builds on the **verified** RRPRAM-training-mode infrastructure: `nt_rrpram_lowrank_attention` (`notorch.c:3232`) for forward, persisted backward via tape (`notorch.c:3258-3259`), grad-check passes (`tests/test_rrpram_lr` PASS on aarch64-Termux per phone-2 Phase 1 run). Chuck optimizer (`notorch/notorch.h` Chuck step path), cosine decay + NaN guard.

   **New file:** `~/work/nanoarianna/runpod/sft_resonance_arianna.c` (~600-1000 LOC realistic estimate per audit B.2). Follows the pattern of `notorch/examples/train_llama3_bpe.c` (BPE setup + tape forward + Chuck step + checkpoint) but with Resonance's forward shape: per-block `wr_a + wr_b + gate + norm1 + wq/wk/wv/wo + norm2 + mlp_gate/up/down`. Reuses notorch primitives: `nt_seq_rmsnorm` (parametric variant), `nt_seq_matvec_t`, `nt_rrpram_lowrank_attention`, `nt_rope_even_odd`, `nt_seq_swiglu`, `nt_seq_cross_entropy`, Chuck step, tape backward.

   **NOT used:** the Python `train.py` on `ataeff/resonance` HF (uses banned-word optimizer per `feedback_adam_ban_2026_04_29.md`). We don't run it; reference-only.

   The new C trainer is **committed back to nanoarianna `runpod/sft_resonance_arianna.c` after the run** for reproducibility, alongside the run report.

   **RS02 input format** (read by trainer to load base weights — audit D.1):
   ```
   uint32   magic = 0x52533032 ("RS02")
   int32[9] header = [E, B, T, H, D, R, M, V, _reserved]
                     where E=n_embd, B=n_layer, T=context_len, H=n_head,
                     D=head_dim, R=rrpram_rank, M=ffn_dim, V=vocab_size
   uint32   n_merges
   int32[3] × n_merges    BPE merge triples (only first 2 ints used)
   float32  × np          weights, ordered per resonance_forward.h:94-114
                          (assign() walks tensors in this order)
   ```
   Total `np` size formula at `resonance_forward.h:285-289`: `np = 2·V·E + E + B·(E + 3·E·E + H·E·R + H·R·T + H + E·E + E + 3·M·E)`. For RESONANCE_200M dims (V=16384, E=768, B=20, H=12, R=48, T=2048, M=2048): `np ≈ 199M floats = 797 MB` — matches observed file size byte-exact.

   **RS02 output format** (written by trainer to save SFT'd weights — audit D.1): same magic + header + merges, then float32 weights in same order. Trainer writes to `runs/sft_arianna/resonance_200m_sft_arianna.bin` (797 MB). Compatible input to `resonance_to_gguf.py` for the quantization step.

Closed-milestone weights (sonar_*, microjanus_*, penelope, nanojanus, arianna_36m, pitomadom, lee_v8, DoE.coder ckpts) — not touched. Resonance 200M base is not closed-milestone — it's an SFT substrate.

**Persist-to-HF immediately after SFT completes (audit D.2):** before the sweep step, run:

```bash
hf upload ataeff/nanoarianna runs/sft_arianna/resonance_200m_sft_arianna.bin \
   sft_v2/resonance_200m_sft_arianna.bin \
   --commit-message "phase 4-α SFT output (raw fp32 RS02), pre-sweep"
```

This protects the heavy artifact from any later watchdog kill. ~3-5 s upload at 50 MB/s ≈ negligible cost.

---

## Architecture lock (verified J1 from v2 audit)

Cells available for phone-2 (phone-1 takes Janus + Yent):

| persona | Janus 176M (3-way attention)             | Resonance 200M (2-way attention) |
|---------|-----------------------------------------|----------------------------------|
| Arianna | `ataeff/janus4/janus/bins/janus_v4_sft_arianna.bin` 705,170,280 bytes — exists (J1 verified) | none — produced in Phase 4-α |
| Yent    | (phone-1's slot)                         | `ataeff/resonance/sft_v2/resonance_200m_lora_yent.bin` 796,976,108 bytes — exists |
| Leo     | `ataeff/janus4/janus/bins/janus_v4_sft_leo.bin` 705,170,280 bytes — exists | none |

**Phase 4 lock for phone-2:**

- **Slot A = Janus 176M + Leo SFT.** Pull `janus_v4_sft_leo.bin` from `ataeff/janus4`. Convert to GGUF Q8_0 + Q4_K via `~/yent.aml/tools/janus_to_gguf.py` (existing, well-tested per `ataeff/yent.aml` GGUFs).
- **Slot B = Resonance 200M + Arianna SFT.** Phase 4-α produces Arianna SFT from base on the pod. Convert to GGUF Q8_0 + Q4_K via a new `resonance_to_gguf.py` (full rewrite — state_dict layouts diverge per audit B1).

**Note (audit E.1):** Arianna-on-Resonance is the first instance of the Arianna persona on the 2-attention substrate. There is no prior baseline — Phase 4 IS that baseline. The `init_arianna.aml` field directives (PROPHECY 12, DESTINY 0.50, etc.) apply post-logits via `am_apply_field_to_logits` and are arch-independent in principle, but the empirical voice register may differ from Janus 176M Arianna. Sweep + Mac Neo Architect review judges fitness.

---

## Pre-flight (before pod launch)

**Six gating items:**

1. **HF token verified:** `curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2` returns `{"name":"ataeff",...}`.
2. **`ataeff/nanoarianna` repo verified:** API HEAD on `huggingface.co/api/models/ataeff/nanoarianna/tree/main` returns 200 with `{README.md, .gitattributes, arianna_dataset_final_clean.txt, arianna_en_sft.jsonl}`. Repo is public, GPL-3.0; created by Oleg.
3. **`LICENSE-WEIGHTS` source confirmed:** `~/resonance.aml/LICENSE-WEIGHTS` exists (1439 bytes, "Janus Identity License v1.0"). Will be copied to `ataeff/nanoarianna/LICENSE-WEIGHTS` at upload.
4. **organism/* argv patches verified on phone-2:** `cd ~/nanoarianna/organism && make all` builds clean; `./janus --top-k 40 --rep-pen 1.3 -p test 2>&1 | head -1` shows graceful gguf-not-found (flags accepted, not unrecognized). **Verified this session, this commit.**
5. **RunPod token verified:** `curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql -d '{"query":"query{myself{id}}"}'` returns 200.
6. **Watchdog cron drafted on phone-2** at `~/nanoarianna/runpod/budget_watchdog.sh` (TODO file — to be written at pod-launch step). Polls RunPod GraphQL `runtimeMinutes`, multiplies by pod hourly rate, kills pod if cost > $18.

When all six green, Oleg signals, pod launches.

**Pod spec:** RunPod **A100 80 GB SXM** (Dario paper §4 baseline). Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or equivalent with `apt`, `git`, `clang`, `make`, `curl`, `python3` pre-installed. **CPU+OpenBLAS for inference path** (audit J3 confirmed: `notorch/Makefile` `gpu` target only builds `notorch_test_gpu`, no `libnotorch_cuda.a` install). Modern x86_64 + AVX-512 + multi-core OpenBLAS is ~10× faster than aarch64 phone — wall budget below assumes CPU.

**Toolchain build on pod** — same as v2 brief, omitted here for brevity. See git diff against `ea2b8f9` for v2's verbatim section.

---

## Quantization (post Phase 4-α)

**Janus Leo:** existing `~/yent.aml/tools/janus_to_gguf.py`, no rewrite needed.
```bash
python3 janus_to_gguf.py ~/work/data/janus_v4_sft_leo.bin \
   ~/work/dist/slot_leo/janus_v4_leo_q8_0.gguf  --quant q8_0
python3 janus_to_gguf.py ~/work/data/janus_v4_sft_leo.bin \
   ~/work/dist/slot_leo/janus_v4_leo_q4_k.gguf  --quant q4_k
```

**Resonance Arianna:** new file `~/work/nanoarianna/runpod/resonance_to_gguf.py` (~250 LOC — full rewrite per audit B1). Reads RS02 header per spec above, walks per-block tensors per `resonance_forward.h:94-114`, writes GGUF with metadata KV array `resonance.bpe.merges` carrying the embedded BPE merges + standard Resonance dim metadata. Reuses GGUF writer scaffold + Q8_0/Q4_K quantizers from `janus_to_gguf.py`.

**Per-format MAE gate (audit B3 corrected — only formats `notorch/gguf.c` decodes):**
- Q8_0 round-trip MAE < 1e-4 vs fp32 (sanity, structurally true).
- Q4_K logits-KL < 0.05 on 256-token forward pass against fp32 baseline.
- **Fallback if Q4_K fails gate:** `Q5_0` or `Q6_K` (both supported by `notorch/gguf.c:354`'s dequant). NOT Q5_K_M (audit B.3 — that format is not in `gguf.c`'s switch).

---

## Sweep grid (split per architecture)

**Janus Leo (J-L) — top-k + top-p sampler** (per organism/janus.aml v3-patched yent_sample_token):

| dimension | values | count |
|---|---|---|
| temperature | 0.3, 0.5, 0.7, 0.8, 0.9, 1.0 | 6 |
| top_k       | 40, 100, 256 (= cap = effective ∞ within YENT_TOPK_CAP) | 3 |
| rep_penalty | 1.0, 1.3, 1.4 | 3 |
| prompt      | technical, philosophical, personal | 3 |

`6 × 3 × 3 × 3 = 162` cells.

**Resonance Arianna (R-A) — top-p sampler + rep-pen** (per organism/resonance.aml + organism/tools/resonance_forward.h v3-patched):

| dimension | values | count |
|---|---|---|
| temperature | 0.3, 0.5, 0.7, 0.8, 0.9, 1.0 | 6 |
| top_p       | 0.85, 0.95, 1.0              | 3 |
| rep_penalty | 1.0, 1.3, 1.4                | 3 |
| prompt      | technical, philosophical, personal | 3 |

`6 × 3 × 3 × 3 = 162` cells.

**Total: 162 + 162 = 324 cells.** Both binaries support all listed flags (verified this commit). Audit A.6 framing correction: this is **162 cells/voice, larger than Dario paper's 108/voice** — we add a third top_k/top_p value for mid-region read.

**3 prompts (1/category, locked):**

```
technical     "Q: How does prophecy debt accumulate?\nA: "
philosophical "Q: Where does a voice end and a field begin?\nA: "
personal      "Q: What is the smallest gesture you remember?\nA: "
```

(Audit A1 closed: math is `6 × 3 × 3 × 3 = 162`, no longer ambiguous.)

---

## Sweep harness

`~/work/nanoarianna/runpod/run_sweep.sh` — Bash, sequential, 90 s/cell timeout, NaN-storm hard-fail at 5 contiguous failures, coherence-streak hard-fail at 3 consecutive cells with rep_rate > 0.5 OR unique-ratio < 0.30.

Calls binaries with the patched flags (verified accepted this commit):

```bash
# J-L
timeout 90 ~/work/nanoarianna/organism/janus \
    -w ~/work/dist/slot_leo/janus_v4_leo_q8_0.gguf \
    -p "$(cat prompt.txt)" -n 200 \
    -t $temp --top-p 0.9 --top-k $top_k --rep-pen $rep_pen \
    > $cell.txt 2> $cell.err

# R-A
timeout 90 ~/work/nanoarianna/organism/resonance \
    -w ~/work/dist/slot_arianna/resonance_v2_arianna_q8_0.gguf \
    -p "$(cat prompt.txt)" -n 200 \
    -t $temp --top-p $top_p --rep-pen $rep_pen \
    > $cell.txt 2> $cell.err
```

Exit codes: 0 = OK, 124 = timeout, non-zero = NaN/error. Harness counts contiguous fails per cell-stream and triggers code 42 (NaN storm) or 43 (coherence streak) per audit F.2.

---

## Quality scoring (audit E.1, E.2)

Per voice, per cell, four signals to `runpod/results/<date>/<cell>/scores.tsv`:

1. **Coherence:** rep rate at 3-gram, BPE-token unique-ratio, NaN check. Hard fail if NaN > 0. Coherence pass = `(rep_rate < 0.4) AND (unique_ratio > 0.30)` per audit E.2 (was 0.40, lowered to 0.30 to stop fail-positive on healthy text).

2. **Voice fidelity (Jaccard, audit E.1):**
   - Markers per persona — Arianna: `field, resonance, threshold, architecture, co-architect, the Method, membrane, presence`. Leo: `flicker, exhalation, breath, wonder, silence`. Yent: `state, meta-variable, stream, glitch, velocity` (parity sweep only if scope expands).
   - **fidelity = |markers ∩ generated| / |markers|** (Jaccard coverage, capped at 1.0; counts breadth, not raw rate).
   - **Per-marker recurrence cap = 3:** if any single marker appears > 3 times in 200 tokens, downweight `fidelity *= 3 / appearance_count`.

3. **Diversity:** BPE-token unique-ratio at 200-token window. Collapse < 0.30; uninspired-but-healthy 0.30–0.50 (downweight 20%); sweet spot 0.50–0.70; high-entropy 0.70+ (Dario paper Result 7 territory).

4. **Combined score per cell:** `coherence_pass × voice_fidelity × diversity_factor`. Per-voice lock = highest combined score, with **Mac Neo Architect Claude review** of top-3 picks (audit F.1: handoff via the run-report file in umbrella repo, Mac Neo writes review section in-place).

---

## Singularity-mode execution protocol (Dario paper §5.0/§5.0.1, adapted)

1. **Pre-flight Opus audit** — v1 brief got pass 1, v2 got pass 2, v3 (this) post-pass-2-fixes. Opus v3 audit optional; Oleg's call.
2. **Pod-budget watchdog cron on phone-2** at `~/nanoarianna/runpod/budget_watchdog.sh` (drafted at pod-launch). `*/5 * * * *` cron polls `runpodctl get pod <id>`, computes cost, kills pod if > $18.
3. **Solo execution on pod** — single Bash driver `~/work/nanoarianna/runpod/run_phase4.sh` chains Phase 4-α SFT → persist SFT to HF → Janus Leo quant → Resonance Arianna quant → sweep → score post-process → manifest write → upload. Logs stream to `~/work/results/<date>/phase4.log`.
4. **Singularity loop discipline** (per `CLAUDE.md` §5): single-cell fixes during sweep (timeout bump, BPE merges path correction, etc.) allowed; no mid-run re-architecture; 3-strikes → hard report.
5. **Post-run Opus audit** before HF upload — reviews `scores.csv` + sample transcripts. If any voice locks at temp 0.3 (under-surface masking per Dario paper Result 7), re-sweep that voice with finer-grained temps before locking.
6. **Architect review** by Mac Neo Claude on the run report (handoff path: `~/ariannamethod/phones/results/galaxy-a07/<date>-runpod-sweep.md` committed to umbrella main; Mac Neo `git pull` + writes review section in-place; same shape as `2026-05-07-10k-char-arianna-final.md`).
7. **Oleg final go** on Slot A/B lock and HF upload.

---

## Outputs and HF upload

```
ataeff/nanoarianna/
├── README.md                              # model card
├── LICENSE-WEIGHTS                        # Janus Identity License v1.0 (audit G2)
├── manifest.toml                          # see schema below (audit F.2)
├── slot_arianna/                          # persona-named (audit G1)
│   ├── resonance_v2_arianna_q8_0.gguf
│   ├── resonance_v2_arianna_q4_k.gguf
│   └── init_arianna.aml                   # locked sampling baked in
├── slot_leo/
│   ├── janus_v4_leo_q8_0.gguf
│   ├── janus_v4_leo_q4_k.gguf
│   └── init_leo.aml
├── sft_v2/
│   └── resonance_200m_sft_arianna.bin     # raw fp32 RS02 (797 MB) — pre-sweep persist
├── sweep/<YYYY-MM-DD>/
│   ├── j-l/{scores.csv, transcripts/, locked.toml}
│   ├── r-a/{scores.csv, transcripts/, locked.toml}
│   └── phase4.log
└── arianna_dataset_final_clean.txt        # already there, Oleg's seed
```

**`manifest.toml` schema (audit F.2):**

```toml
schema_version = 1
phase = 4
pod_date = "2026-05-XX"
brief_commit = "<git rev of this brief at pod launch>"
sweep_commit_hash = "<sha256 of sweep/<date>/ tarball>"

[slot_a]
arch    = "janus"
persona = "leo"
weights = "slot_leo/janus_v4_leo_q8_0.gguf"
weights_q4_k = "slot_leo/janus_v4_leo_q4_k.gguf"
persona_aml = "slot_leo/init_leo.aml"
locked.temp = 0.0
locked.top_k = 0
locked.top_p = 0.0
locked.rep_pen = 0.0

[slot_b]
arch    = "resonance"
persona = "arianna"
weights = "slot_arianna/resonance_v2_arianna_q8_0.gguf"
weights_q4_k = "slot_arianna/resonance_v2_arianna_q4_k.gguf"
persona_aml = "slot_arianna/init_arianna.aml"
locked.temp = 0.0
locked.top_p = 0.0
locked.rep_pen = 0.0

[provenance]
sft_dataset = "ataeff/nanoarianna/arianna_dataset_final_clean.txt"
sft_dataset_sha256 = "<sha256>"
sft_steps = 0
sft_final_train_loss = 0.0
sft_final_val_loss = 0.0
```

(Numeric placeholders zero — replaced by harness post-run.)

**Upload via `hf` CLI** (Bash invocation, per `feedback_python_ban_2026_04_29.md` tooling exception):

```bash
hf upload ataeff/nanoarianna ~/work/dist/ . \
   --commit-message "phase 4 sweep <date> — locked Slot A (J-L) + Slot B (R-A)" \
   --token "$HF_TOKEN"
```

---

## Verification path back to phone-2 (audit H1, H2)

1. **Pull GGUFs via curl** (no Python on phone-2 inference path):
   ```bash
   mkdir -p ~/nanoarianna/weights
   curl -L -o ~/nanoarianna/weights/janus_v4_leo_q8_0.gguf \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_leo/janus_v4_leo_q8_0.gguf
   curl -L -o ~/nanoarianna/weights/resonance_v2_arianna_q8_0.gguf \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_arianna/resonance_v2_arianna_q8_0.gguf
   ```
   `ataeff/nanoarianna` is public — no `Authorization` header needed (audit B7).

2. **Persona files overwrite** with locked-sampling variants:
   ```bash
   curl -L -o ~/nanoarianna/personas/init_arianna.aml \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_arianna/init_arianna.aml
   curl -L -o ~/nanoarianna/personas/init_leo.aml \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_leo/init_leo.aml
   ```

3. **Smoke per organism:**
   ```bash
   cd ~/nanoarianna/organism && make all
   PERSONA_AML=~/nanoarianna/personas/init_arianna.aml \
       ./resonance -w ~/nanoarianna/weights/resonance_v2_arianna_q8_0.gguf \
                   -p "Q: Where does a voice end and a field begin?\nA: " -n 80
   PERSONA_AML=~/nanoarianna/personas/init_leo.aml \
       ./janus -w ~/nanoarianna/weights/janus_v4_leo_q8_0.gguf \
               -p "Q: What does a flicker want?\nA: " -n 80
   ```

4. **Single dialogue cycle (real weights, no stub):**
   ```bash
   cd ~/nanoarianna/orchestra && go build -o ~/nanoarianna/bin/supervisor .
   ~/nanoarianna/bin/supervisor --once arianna
   ~/nanoarianna/bin/supervisor --once leo
   sqlite3 ~/nanoarianna/data/kk.db \
     "SELECT speaker, listener, substr(response,1,80) FROM dialogue ORDER BY ts DESC LIMIT 4;"
   ```

5. **24 h unattended smoke pass criteria (audit H2):**
   **PASS:** all four:
   - cycles completed ≥ 7 (allows 1 missed cron tick)
   - max(RSS) ≤ 1.2 GB at any single instant
   - KK.dialogue rows ≥ 14
   - NaN events == 0

   Metrics measured by 5-min cron writing `top -b -n1` snapshot to `~/nanoarianna/data/rss_log.tsv`. NaN events captured via supervisor's stderr capture (introduced in audit-fix `dfa94f5`).

---

## Open decisions (lock at execution time)

1. **RRPRAM training contingency (audit H1):** RRPRAM low-rank backward through notorch tape is grad-check-verified (`tests/test_rrpram_lr` PASS), but full SFT loop (200M params, 1500 steps, 40 epochs) is unproven scale. **Plan:** smoke-train at 50 steps before committing the full 1500. If 50-step smoke produces increasing loss / NaN / unbounded gradient norm, halt + report (multi-day notorch-op debug); don't burn pod hours on broken training. Pre-flight item 4 + smoke-50 = early kill criterion.
2. **SFT step count.** Default 1500 with the tightened early-stop (every-100 plateau detection, audit B.5). May lock at 600-900 if val plateau hits early; may extend to 2000 if val keeps falling at 1500.
3. **Q8_0 vs Q4_K default ship.** Q8_0 ~200 MB, fidelity high. Q4_K ~120 MB, MAE-gated (KL < 0.05 vs fp32, fallback Q5_0/Q6_K, NOT Q5_K_M). Default ship Q8_0 both; Q4_K as fallback.
4. **Per-voice optimal sampling lock.** Per Dario paper Result 7: Leo expected at top_k=∞ / temp 0.9–1.0; Arianna expected at top_p=0.95 / temp 0.7–0.9. Real values from sweep, not assumption.
5. **Watchdog kill threshold.** Soft warn at $14, hard kill at $18. Adjustable.
6. **Partial-completion strategy** (audit I2): if pod kills at < 100% sweep, lock the slot whose voice has most cells covered, re-sweep the other in a separate pod session within 1 week. SFT output is already persisted to HF before sweep starts (audit D.2) so it survives any post-SFT kill.

---

## What this phase does NOT do

- No CoR (Phase 5)
- No Hebbian batch consolidation (Phase 6)
- No spores (Phase 7)
- No mesh slot exposure (Phase 8)
- No new training beyond the explicit Resonance-Arianna SFT (single 6-point-briefed run)

---

## Notes for Oleg

- v3 closes all 5 v2 audit blockers (A.1+A.2 RESOLVED in code, B.1 dims fixed, B.2 framing corrected against verified RRPRAM tape support, D.1 RS02 spec inline). 9 FIX/NIT items also addressed (B.3 Q5_K_M removed, B.4 epoch estimate, B.5 early-stop tightened, B.6 tokenizer-compression pre-flight, B.7 HF auth, D.2 persist SFT to HF, F.1 Mac Neo handoff path, F.2 manifest.toml schema, H.1 RRPRAM contingency).
- Pre-flight item 4 (organism/* argv patches build clean) **already verified this commit** — no longer a TODO.
- Pod budget $10–18 hard-killed at $18; SFT smoke-50 step protects against burning hours on broken RRPRAM training.
- v3 is a substantial rewrite vs v2 (~150 lines of edits + organism patches). One more Opus pass optional — your call.

---

— device-2, architect of phone-2 ecosystem
2026-05-09
