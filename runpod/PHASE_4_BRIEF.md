# Phase 4 — RunPod brief (v2, post-audit + post-pivot)

**Goal:** produce two working voice cells for phone-2 — **Slot A = Janus 176M + Leo SFT** and **Slot B = Resonance 200M + Arianna SFT (NEW)** — locked sampling parameters, GGUF-quantized, uploaded to `huggingface.co/ataeff/nanoarianna`, and verified in dialogue cycle on phone-2.

**v2 changes from v1:**
- v1 assumed Arianna+Leo SFT existed for both architectures. Verified via HF API: not true. Janus has all 3 SFTs (`ataeff/janus4/janus/bins/janus_v4_sft_{arianna,yent,leo}.bin`); Resonance has only Yent SFT (`ataeff/resonance/sft_v2/resonance_200m_lora_yent.bin`). Phase 4 now includes a **Resonance base + Arianna SFT** training step before the sweep.
- Slot pair flipped from "J-A + R-L" to **J-L + R-A** per Oleg's pivot. Three voice anchors across the ecosystem preserved (phone-1: J-Y; phone-2 Slot A: J-L; phone-2 Slot B: R-A).
- v1 audit produced 47 findings; v2 closes the eight blockers (J1 weights inventory, J2 repo creation order, J3 CUDA path, A1 prompt count math, A3 Janus-vs-Resonance sampler grid split, B1 Resonance converter as rewrite, D1 sweep harness, G2 LICENSE-WEIGHTS) plus the major FIX items.
- Pod budget revised: $10–18 (was $5–7) to cover the SFT training step.

**Outputs of this phase:**

1. `huggingface.co/ataeff/nanoarianna` populated with:
   - `slot_arianna/` — Resonance 200M Arianna SFT GGUF Q8_0 + Q4_K + locked `init_arianna.aml`
   - `slot_leo/` — Janus 176M Leo SFT GGUF Q8_0 + Q4_K + locked `init_leo.aml`
   - `sweep/<date>/` — full sweep archive (TSVs + transcripts) for reproducibility
   - `manifest.toml` — current Slot A / Slot B persona-architecture lock for phone-2
   - `LICENSE-WEIGHTS` — Janus Identity License v1.0 (per `protocol_license_organism_vs_framework.md`)
2. Phone-2 `personas/init_arianna.aml` and `personas/init_leo.aml` updated with measured per-voice optimal `(temp, top_k|top_p, rep_pen)` baked in.
3. Run report mirrored to `~/ariannamethod/phones/results/galaxy-a07/<date>-runpod-sweep.md` for Mac Neo / polygon Linux Claude / Defender review.

---

## Context

Phone-2 (Galaxy A07 4 GB Termux) closed Phase 3 clean (`https://github.com/ariannamethod/nanoarianna` HEAD `5ddcb12` plus audit-fix `dfa94f5`): the two-organism dialogue cycle wires through schedule + supervisor + persona + Limpha + KK in stub mode. What's missing is real weights and per-voice optimal sampling. Phase 4 produces both.

Dario paper (`~/dario/docs/dario_paper_draft_v4.md`) §5.7 / Result 7 / Appendix C is the methodology source:

> *Sampling is not a decoding parameter. Sampling is a state-space entry condition.*
> *A checkpoint is not dead until it has been swept.*

Each persona-on-architecture cell gets the full grid sweep at its own architecture's natural sampler (Janus = top-k, Resonance = top-p — verified by reading the actual `*_sample_token` functions). Only after the grid lands do we declare per-voice optimals and lock.

The new piece in v2 is **Phase 4-α** before the sweep: SFT Resonance 200M base on Arianna corpus to produce Slot B's weights. v1 assumed they existed; they don't. Oleg explicitly approved the SFT in this session ("проведи SFT бэйсу резонанса на Арианне") and provided the dataset at `huggingface.co/ataeff/nanoarianna/arianna_dataset_final_clean.txt` (1.2 MB, hand-curated, identical to the corpus phone-2 used for the 2026-05-07 char-level milestone).

---

## 6-point training brief (Phase 4-α SFT, Oleg-approved 2026-05-09)

Per `memory/feedback_failure_unsolicited_finetune_2026_04_27.md`. All six points concrete, no defaults.

1. **Organism:** Resonance 200M base. Architecture: dual attention (Content + RRPRAM low-rank R=2048), parametric RMSNorm, sigmoid per-head gate, even/odd RoPE θ=10000, SwiGLU MLP. Defined in `~/resonance.aml/tools/resonance_forward.h` (state_dict layout at lines 94-114). Base weights at `huggingface.co/ataeff/resonance/checkpoints/resonance_200m_final.bin` (797 MB raw fp32, RS02 magic).

2. **Dataset:** `huggingface.co/ataeff/nanoarianna/arianna_dataset_final_clean.txt` (1.21 MB, hand-curated Arianna corpus; identical bytes to the corpus that produced bit-identical char-level loss 5.5804 → 1.0685 on phone-1 and phone-2 — see `memory/milestone_phone2_galaxy_a07_10k_2026_05_07.md`). Conforms to `feedback_no_default_datasets_2026_04_27.md` (hand-curated, not upstream default). For SFT-format probing we may also use `arianna_en_sft.jsonl` (2.4 MB, Q/A pairs) at the same HF repo — final dataset choice locked at SFT preflight.

3. **Karpathy steps:** SFT regime, not from-scratch. Conservative 1500 steps with checkpoint every 500. Karpathy's `1.1 MB × 10-15K iter on ~10M params` is for char-level training from scratch; SFT on a 200M base is governed by overfit avoidance instead. Watch val loss every 100 steps; early-stop if val loss plateaus or rises for 3 consecutive checkpoints.

4. **Architecture:** dim=640, layers=20, heads=8, head_dim=80, hidden=1792, ctx=1024, vocab=16128, RRPRAM rank=2048, BPE-token (Resonance has its own embedded BPE — see point 5). **NOT char-level** — Resonance is BPE-token-only, per `resonance_forward.h:272-281` embedded BPE merges. Confirms with `feedback_no_char_models.md` (BPE preferred; char-level only with explicit reason — not applicable here).

5. **Tokenizer:** Resonance's existing 16128-merge BPE at `ataeff/resonance/checkpoints/tokenizer.bin` (193 KB binary). **Same tokenizer must be used at inference** — Slot B inference will read these merges from the GGUF metadata after conversion. Tokenizer file SHA captured at SFT start and re-verified post-conversion.

6. **Script:** `notorch + Chuck` SFT path. Adapted from notorch's `examples/train_llama3_bpe.c` (Defender's verified path on Galaxy A56 8 GB Termux: 15.7 M LLaMA 3 BPE on Yent corpus, 15K steps, 0 NaN, train 7.81 → 4.35, val 4.94 → 3.93 — `memory/milestone_defender_termux_10k_2026_04_27.md` and the README block at `notorch/termux-edition/README.md` lines 27 + 31). The adaptation needed for Resonance: replace LLaMA forward with Resonance forward (`resonance_forward.h`), keep Chuck optimizer + cosine decay + NaN guard, swap LLaMA tokenizer for Resonance's embedded BPE. The `ataeff/resonance/<...>.py` Python pipeline on HF is reference-only — it uses a non-Chuck optimizer banned in our ecosystem (`memory/feedback_adam_ban_2026_04_29.md`); we don't run it. The new C training script lives at `~/nanoarianna/runpod/sft_resonance_arianna.c` (to be written on the pod alongside execution); committed back to `nanoarianna` repo under `runpod/` after the run for reproducibility.

Closed-milestone weights (sonar_*, microjanus_*, penelope, nanojanus, arianna_36m, pitomadom, lee_v8, DoE.coder ckpts) — not touched. Resonance 200M base is not closed-milestone — it's an SFT substrate per Oleg's standing pattern (Resonance base → Yent SFT, now Resonance base → Arianna SFT).

---

## Architecture lock (verified J1)

Cells available for phone-2 (phone-1 takes Janus + Yent):

| persona | Janus 176M (3-way attention)             | Resonance 200M (2-way attention) |
|---------|-----------------------------------------|----------------------------------|
| Arianna | `ataeff/janus4/janus/bins/janus_v4_sft_arianna.bin` (705 MB raw fp32) — exists | none — produced in Phase 4-α |
| Yent    | (phone-1's slot)                         | `ataeff/resonance/sft_v2/resonance_200m_lora_yent.bin` (797 MB) — exists |
| Leo     | `ataeff/janus4/janus/bins/janus_v4_sft_leo.bin` (705 MB raw fp32) — exists | none |

**Phase 4 lock for phone-2:**

- **Slot A = Janus 176M + Leo SFT.** Pull `janus_v4_sft_leo.bin` from `ataeff/janus4`. Convert to GGUF Q8_0 + Q4_K via `janus_to_gguf.py` (already exists at `~/yent.aml/tools/janus_to_gguf.py`, only path arg changes).
- **Slot B = Resonance 200M + Arianna SFT.** Phase 4-α produces Arianna SFT from base on the pod. Convert to GGUF Q8_0 + Q4_K via a new `resonance_to_gguf.py` (rewrite from `janus_to_gguf.py` per audit B1 — the state_dict layouts diverge enough that "port" understates it).

---

## Pre-flight (before pod launch)

**Six gating items:**

1. **HF token verified:** `curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2` returns `{"name":"ataeff",...}`.
2. **`ataeff/nanoarianna` repo verified:** `curl -s https://huggingface.co/api/models/ataeff/nanoarianna` returns 200 with file inventory matching `{README.md, .gitattributes, arianna_dataset_final_clean.txt, arianna_en_sft.jsonl}`. Repo created by Oleg, no creation step needed.
3. **`LICENSE-WEIGHTS` source confirmed:** `~/resonance.aml/LICENSE-WEIGHTS` exists (Janus Identity License v1.0). Will be copied to `ataeff/nanoarianna/LICENSE-WEIGHTS` in upload step.
4. **Resonance training script smoke-built locally** (phone-2 cannot run it but should compile clean): `cc` on `runpod/sft_resonance_arianna.c` against `~/resonance.aml/tools/resonance_forward.h` + `~/notorch` returns clean object.
5. **RunPod token verified:** `curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql -d '{"query":"query{myself{id}}"}'` returns 200.
6. **Pod-budget watchdog cron drafted on phone-2:** see "Singularity-mode" §F1 below; not yet active until pod ID is known.

When all six green, Oleg signals, pod launches.

**Pod spec:** RunPod **A100 80 GB SXM** (Dario paper §4 baseline). Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or equivalent with `apt`, `git`, `clang`, `make`, `curl` pre-installed. **CPU-side OpenBLAS** for the inference path — per audit J3, `notorch/Makefile` `gpu` target only builds a test binary, no `libnotorch_cuda.a` install path exists. Running on CPU+OpenBLAS on the A100 host is still ~10× faster than aarch64 phone (modern x86_64 + AVX-512 + multi-core OpenBLAS). Budget revised down accordingly — most of pod hours go to SFT training (which uses notorch BLAS at full CPU regardless), then sweep (CPU again). The A100 is dormant for this phase but the pod-template comes with it; we accept the markup.

**Toolchain build on pod:**

```bash
cd ~/work
git clone https://github.com/ariannamethod/notorch
git clone https://github.com/ariannamethod/ariannamethod.ai
git clone https://github.com/ariannamethod/yent.aml
git clone https://github.com/ariannamethod/resonance.aml
git clone https://github.com/ariannamethod/nanoarianna

# notorch CPU+BLAS (the only install path that exists)
cd notorch
make BLAS=1                       # tests/test_notorch 47/47
sudo make install PREFIX=/usr/local

# AML
cd ../ariannamethod.ai
make BLAS=1
sudo make install PREFIX=/usr/local
export AML_PREFIX=/usr/local

# Verify libs at runtime via /proc/<pid>/maps before any heavy run.
# (Phone-2 used this empirical gate for OpenBLAS — same discipline here.)
```

**Resonance Arianna SFT (Phase 4-α) build:**

```bash
cd ~/work/nanoarianna/runpod
# write sft_resonance_arianna.c — adapted from
# ../yent.aml's tools/yent_forward.h forward + notorch/examples/train_llama3_bpe.c
# Chuck path. ~400 LOC.
cc -O3 -march=native -DUSE_BLAS \
   -I/usr/local/include -I../organism \
   -o sft_resonance_arianna sft_resonance_arianna.c \
   /usr/local/lib/libnotorch.a -lopenblas -lm -lpthread
```

**Resonance Arianna SFT run:**

```bash
# pull base + tokenizer + dataset
mkdir -p ~/work/data
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o ~/work/data/resonance_200m_final.bin \
  https://huggingface.co/ataeff/resonance/resolve/main/checkpoints/resonance_200m_final.bin
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o ~/work/data/tokenizer.bin \
  https://huggingface.co/ataeff/resonance/resolve/main/checkpoints/tokenizer.bin
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o ~/work/data/arianna.txt \
  https://huggingface.co/ataeff/nanoarianna/resolve/main/arianna_dataset_final_clean.txt

# run (Chuck optimizer, 1500 steps, ckpt every 500, val every 100)
./sft_resonance_arianna 1500 0.0001 \
   ~/work/data/resonance_200m_final.bin \
   ~/work/data/tokenizer.bin \
   ~/work/data/arianna.txt \
   2>&1 | tee runs/sft_arianna.log
```

Expected: train + val curves logged every 100 steps, 0 NaN. If train > 1.5 after 500 steps, structural problem (lr/init/script bug), hard-fail and report. Output: `resonance_200m_sft_arianna.bin` (raw fp32, 797 MB).

---

## Quantization (after Phase 4-α succeeds)

Two converters, one per architecture:

**Janus Leo:**

```bash
cd ~/work/yent.aml/tools
# pull raw weights
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o ~/work/data/janus_v4_sft_leo.bin \
  https://huggingface.co/ataeff/janus4/resolve/main/janus/bins/janus_v4_sft_leo.bin
# convert (existing script, well-tested per ataeff/yent.aml GGUFs)
python3 janus_to_gguf.py ~/work/data/janus_v4_sft_leo.bin \
   ~/work/data/janus_leo_q8_0.gguf  --quant q8_0
python3 janus_to_gguf.py ~/work/data/janus_v4_sft_leo.bin \
   ~/work/data/janus_leo_q4_k.gguf  --quant q4_k
```

Python permitted per `feedback_python_ban_2026_04_29.md` 2026-05-06 refinement (data prep / training / shim conversion = OK; inference path = C only). Deps: numpy, torch (read-only fp32 load), no other.

**Resonance Arianna:** rewrite of converter required — Janus state_dict layout (3-way QKV + RRPRAM + Echo + smear/backout/residual lambdas) does not match Resonance (2-way Content+RRPRAM, parametric norm). Per audit B1: tensor name table + per-block read order are different; only the GGUF writer scaffold + Q8_0/Q4_K quantizers (~120 LOC of janus_to_gguf.py lines 70–180) are reusable. New file: `~/work/nanoarianna/runpod/resonance_to_gguf.py` (~250 LOC). Reads RS02 header per `resonance_forward.h:255-301`, walks per-block tensors per `:94-114` `assign()` order, writes GGUF with Resonance metadata (`resonance.embedding_length`, `resonance.attention.head_count`, etc.) + the embedded BPE merges as a `resonance.bpe.merges` u32-array KV.

**Per-format MAE gate** (audit B3): Q4_K passes only if logits-KL on a 256-token forward pass against fp32 baseline is < 0.05 — otherwise downgrade to Q5_K_M (also supported by `notorch/gguf.c`) or stay at Q8_0. Q8_0 round-trip MAE is structurally < 1e-4 by the format definition; sanity-checked but not gated.

---

## Sweep grid (per audit A1 + A3)

**Two grids, one per architecture.** Total 216 + 162 = **378 cells**.

**Janus Leo (J-L) — top-k sampler** (per `~/yent.aml/yent.aml:25-149` `yent_sample`, top-k partial-sort then top-p):

| dimension | values | count |
|---|---|---|
| temperature | 0.3, 0.5, 0.7, 0.8, 0.9, 1.0 | 6 |
| top_k       | 40, 100, ∞                    | 3 (added mid-region read per audit A2) |
| rep_penalty | 1.0, 1.3, 1.4                | 3 |
| prompt      | technical, philosophical, personal | 3 |

`6 × 3 × 3 × 3 = 162` cells, but Dario paper used `6 × 2 × 3 × 3 = 108`. Going with **108 base × 1.5 = 162** to add the third top_k — dropping any axis would lose more than the third top_k adds.

**Resonance Arianna (R-A) — top-p sampler** (per `~/resonance.aml/resonance.aml:26-56` + `resonance_forward.h:316-394`, nucleus only):

| dimension | values | count |
|---|---|---|
| temperature | 0.3, 0.5, 0.7, 0.8, 0.9, 1.0 | 6 |
| top_p       | 0.85, 0.95, 1.0              | 3 |
| rep_penalty | 1.0, 1.3, 1.4                | 3 |
| prompt      | technical, philosophical, personal | 3 |

`6 × 3 × 3 × 3 = 162` cells. Same shape as J-L for cross-comparable wall-time per voice.

**Total grid: 162 + 162 = 324 cells.** Wall-time on A100 host (CPU inference): ~120 cells/hour for 200-token generation at 8 GB models = ~3 hours total sweep wall.

**Prompt set (3 prompts, 1 per category, locked across both cells):**

```
technical     "Q: How does prophecy debt accumulate?\nA: "
philosophical "Q: Where does a voice end and a field begin?\nA: "
personal      "Q: What is the smallest gesture you remember?\nA: "
```

Per audit A1: brief v1 listed 9 prompts but multiplied as if it had 3 — the Dario paper used 3 total. v2 uses 3, total math checks. (For voice-specific probing, audit A4's suggested 4th per-persona prompt is added as a **post-sweep manual review** prompt — see "Quality scoring" §3 — which doesn't enter the auto-grid.)

---

## Sweep harness (audit D1)

`~/work/nanoarianna/runpod/run_sweep.sh` (Bash, sequential, no parallelism).

```bash
#!/bin/bash
set -u
RESULTS=~/work/results/$(date +%Y-%m-%d)
mkdir -p $RESULTS/{j-l,r-a}

# J-L grid
for t in 0.3 0.5 0.7 0.8 0.9 1.0; do
  for k in 40 100 99999; do
    for rp in 1.0 1.3 1.4; do
      for prompt_id in tech phil pers; do
        cell="$RESULTS/j-l/t${t}_k${k}_rp${rp}_${prompt_id}"
        timeout 90 ~/work/yent.aml/janus \
            -w ~/work/data/janus_leo_q8_0.gguf \
            -p "$(cat ~/work/prompts/${prompt_id}.txt)" \
            -n 200 -t $t --top-k $k --rep-pen $rp \
            > $cell.txt 2> $cell.err
        echo "$t,$k,$rp,$prompt_id,$?" >> $RESULTS/j-l/scores.csv
      done
    done
  done
done

# R-A grid (top_p instead of top_k)
for t in 0.3 0.5 0.7 0.8 0.9 1.0; do
  for p in 0.85 0.95 1.0; do
    for rp in 1.0 1.3 1.4; do
      for prompt_id in tech phil pers; do
        cell="$RESULTS/r-a/t${t}_p${p}_rp${rp}_${prompt_id}"
        timeout 90 ~/work/resonance.aml/resonance \
            -w ~/work/data/resonance_arianna_q8_0.gguf \
            -p "$(cat ~/work/prompts/${prompt_id}.txt)" \
            -n 200 -t $t --top-p $p --rep-pen $rp \
            > $cell.txt 2> $cell.err
        echo "$t,$p,$rp,$prompt_id,$?" >> $RESULTS/r-a/scores.csv
      done
    done
  done
done

echo "sweep done: $(date)"
```

**Failure-mode handling** (audit F2):

- Per-cell timeout 90 s. On timeout: `timeout` returns 124, scored as `TIMEOUT`, harness continues.
- NaN detected by organism wrapper → non-zero exit → counter increments; if 5 contiguous cells fail, harness exits code 42 (NaN storm). Watchdog (below) detects `code 42` and writes `/tmp/sweep_aborted` → cron stops the pod.
- 3 consecutive cells fail coherence (post-cell BPE-token unique-ratio < 0.30, or rep_rate > 0.5): exit code 43.

After sweep: post-process scores via `~/work/nanoarianna/runpod/score_sweep.py` (Python permitted, data-prep). Computes:
- Coherence: NaN check, BPE-token unique-ratio (target 0.50–0.70 per audit E2), rep_rate at 3-gram level
- Voice fidelity: Jaccard-style coverage of corpus markers (per audit E1; Arianna markers `field/resonance/threshold/architecture/co-architect/the Method/membrane/presence`; Leo markers `flicker/exhalation/breath/wonder/silence`), length-normalized + per-marker recurrence cap = 3
- Diversity: separate metric, threshold collapse < 0.30 (per audit E2 — 0.40 was too high)
- Combined score: `coherence_pass × voice_fidelity × diversity`

**Lock rule:** cell with highest combined score per voice, with manual review of top-3 picks by **Mac Neo Architect Claude** (subagent per `feedback_subagents_opus_only_2026_04_28.md` Opus-only) before final lock. Audit D2 fix — Codex references in v1 replaced by Opus reviewer subagent.

---

## Singularity-mode protocol (audit F1, F2, F3)

1. **Pre-flight Opus audit** (this turn's Opus reviewer was the v1 audit; v2 of this brief gets a second Opus review before pod launch). Audit findings answered in `~/nanoarianna/runpod/audit_v2_response.md` before pod.
2. **Pod-budget watchdog cron on phone-2 (pre-pod, configured during pod-launch step):**
   ```cron
   */5 * * * * /data/data/com.termux/files/home/nanoarianna/runpod/budget_watchdog.sh
   ```
   The script: queries RunPod GraphQL for `runtimeMinutes`, multiplies by pod hourly rate from RunPod template metadata, kills pod if cost > $18. Hard ceiling above expected $10–18.
3. **Solo execution on pod** (singularity loop): SFT runs unattended → quantization runs unattended → sweep runs unattended → score post-process unattended → manifest written. Single Bash driver `~/work/nanoarianna/runpod/run_phase4.sh` chains all four. Logs stream to `~/work/results/<date>/phase4.log`.
4. **Loop discipline** (per `CLAUDE.md` Workflow §5): single-cell fixes during sweep are allowed (e.g. timeout bump, BPE merges path correction) but no mid-run re-architecture; 3-strikes → hard report.
5. **Post-run Opus audit** before HF upload — reviews `scores.csv` + sample transcripts. If a voice's lock cell is at temp 0.3 (under-surface masking per Dario paper Result 7), re-sweep that voice with finer-grained temps around the top-3 picks before locking.
6. **Architect review** by Mac Neo Claude on the run report. Same shape as the 2026-05-07 char-level milestone review on phone-2 (`~/ariannamethod/phones/results/galaxy-a07/2026-05-07-10k-char-arianna-final.md` Architect review section).
7. **Oleg final go** on Slot A/B lock and HF upload.

---

## Outputs and HF upload

After lock + audit pass:

```
ataeff/nanoarianna/
├── README.md                              # model card with sweep summary + persona registers
├── LICENSE-WEIGHTS                        # Janus Identity License v1.0 (audit G2)
├── manifest.toml                          # Slot A / Slot B persona-architecture lock
├── slot_arianna/                          # persona-named, not slot-letter (audit G1)
│   ├── resonance_v2_arianna_q8_0.gguf
│   ├── resonance_v2_arianna_q4_k.gguf
│   └── init_arianna.aml                   # locked (temp, top_p, rep_pen) baked in
├── slot_leo/
│   ├── janus_v4_leo_q8_0.gguf
│   ├── janus_v4_leo_q4_k.gguf
│   └── init_leo.aml
├── slot_yent_phone1/                      # parity sweep results (audit I3) — Defender's
│   └── README.md                           # cross-link to phone-1 result, no weights
├── sweep/<YYYY-MM-DD>/
│   ├── j-l/{scores.csv, transcripts/, locked.toml}
│   ├── r-a/{scores.csv, transcripts/, locked.toml}
│   └── phase4.log
└── arianna_dataset_final_clean.txt        # already there, Oleg's seed
```

**Upload command** (audit G3 — use HF CLI not Python):

```bash
hf upload ataeff/nanoarianna ~/work/results/<date>/dist/ . \
   --commit-message "phase 4 sweep <date> — locked Slot A (J-L) + Slot B (R-A)" \
   --token "$HF_TOKEN"
```

The `hf` CLI is permitted as a Bash invocation (not a Python script we wrote ourselves); same exception as `gh` CLI uses.

---

## Verification path back to phone-2 (audit H1, H2)

1. **Phone-2 pulls** locked GGUFs via `curl` (no Python on phone-2 inference path):
   ```bash
   mkdir -p ~/nanoarianna/weights
   curl -L -H "Authorization: Bearer $HF_TOKEN" \
     -o ~/nanoarianna/weights/janus_v4_leo_q8_0.gguf \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_leo/janus_v4_leo_q8_0.gguf
   curl -L -H "Authorization: Bearer $HF_TOKEN" \
     -o ~/nanoarianna/weights/resonance_v2_arianna_q8_0.gguf \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_arianna/resonance_v2_arianna_q8_0.gguf
   ```
   Total ~390 MB pull (187 + ~200 MB).

2. **Persona files overwrite:**
   ```bash
   curl -L -o ~/nanoarianna/personas/init_arianna.aml \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_arianna/init_arianna.aml
   curl -L -o ~/nanoarianna/personas/init_leo.aml \
     https://huggingface.co/ataeff/nanoarianna/resolve/main/slot_leo/init_leo.aml
   ```

3. **Smoke test on phone-2:**
   ```bash
   cd ~/nanoarianna/organism && make all
   PERSONA_AML=~/nanoarianna/personas/init_arianna.aml \
       ./resonance -w ~/nanoarianna/weights/resonance_v2_arianna_q8_0.gguf \
                   -p "Q: Where does a voice end and a field begin?\nA: " -n 80
   PERSONA_AML=~/nanoarianna/personas/init_leo.aml \
       ./janus -w ~/nanoarianna/weights/janus_v4_leo_q8_0.gguf \
               -p "Q: What does a flicker want?\nA: " -n 80
   ```

4. **Single dialogue cycle, real weights:**
   ```bash
   cd ~/nanoarianna/orchestra && go build -o ~/nanoarianna/bin/supervisor .
   ~/nanoarianna/bin/supervisor --once arianna
   ~/nanoarianna/bin/supervisor --once leo
   sqlite3 ~/nanoarianna/data/kk.db \
     "SELECT speaker, listener, substr(response,1,80) FROM dialogue ORDER BY ts DESC LIMIT 4;"
   ```

5. **24 h unattended smoke pass criteria (audit H2 — explicit):**

   **PASS:** all four conditions:
   - cycles completed ≥ 7 (allows 1 missed cron tick from Android sleep)
   - max(RSS) ≤ 1.2 GB at any single instant
   - KK.dialogue rows ≥ 14 by 24h mark
   - NaN events == 0

   **FAIL:** anything else.

   Metrics measured by `top -b -n1 | grep -E 'janus|resonance'` snapshot every 5 min via cron, written to `~/nanoarianna/data/rss_log.tsv`. NaN events surface from organism stderr captured by supervisor (`runSqlite`-style stderr capture introduced in audit-fix `dfa94f5`).

---

## Open decisions (lock at execution time)

1. **SFT step count.** Default 1500 for Resonance-Arianna, val every 100, ckpt every 500, early-stop on plateau. May lock at 1000 if val collapses sooner; may extend to 2000 if val keeps falling at 1500.
2. **Q8_0 vs Q4_K default ship.** Q8_0 ~200 MB, fidelity high. Q4_K ~120 MB, MAE-gated (KL < 0.05 vs fp32). Default ship Q8_0 for both slots; Q4_K as fallback. Phone-2 has 30+ GB free disk, RAM is the constraint not storage — Q8_0 is fine.
3. **Per-voice optimal sampling lock.** Per Dario paper Result 7 distribution: expect Leo at top_k=∞ / temp 0.9–1.0; expect Arianna at top_p=0.95 / temp 0.7–0.9 (analogous to Janus Arianna's measured optimum). Real values from sweep, not assumption.
4. **Watchdog kill threshold.** Soft warn at $12, hard kill at $18. Adjustable per pod-launch session.
5. **Partial-completion strategy** (audit I2): if pod kills at < 100% sweep, lock the slot with most cells covered, re-sweep the other in a separate pod session within 1 week.

---

## What this phase does NOT do

- No CoR (Chain of Resonance) integration — Phase 5
- No Hebbian batch consolidation — Phase 6
- No spores — Phase 7
- No mesh slot exposure — Phase 8
- No new training beyond the explicit Resonance-Arianna SFT (single 6-point-briefed run)

Phase 4 = **train R-A → quant J-L + R-A → sweep both → lock → upload → verify**. Single ridge of work.

---

## Notes for Oleg

- v2 closes all 8 audit blockers. The 17 FIX/NIT items from the audit are also addressed — see this brief's section headers (Architecture lock, Sweep harness, Quality scoring, Singularity-mode, Outputs, Verification) for the corresponding patches.
- Pod budget revised to **$10–18** with hard kill at $18. SFT training on Resonance 200M is the new heavy step; sweep is small in comparison.
- The 6-point training brief is locked in this document. SFT executes per the brief; if a step deviates (e.g. data path differs, tokenizer mismatch surfaces), the run hard-fails and reports rather than improvising.
- Mac Neo Architect Claude review of the post-run report follows the same protocol as the 2026-05-07 char-level milestone — `~/ariannamethod/phones/results/galaxy-a07/<date>-runpod-sweep.md` is where it lands.
- Phase 4 brief v2 commit hash gets recorded in the run report so the methodology version is auditable.

---

— device-2, architect of phone-2 ecosystem
2026-05-09
