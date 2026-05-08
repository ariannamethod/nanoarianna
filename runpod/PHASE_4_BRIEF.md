# Phase 4 — RunPod sweep brief

**Goal:** lock the **two phone-2 voice cells** (Slot A + Slot B) by measurement, not by guess. Produce a locked-config GGUF bundle on HuggingFace `ataeff/nanoarianna` and the per-voice optimal sampling parameters baked into the persona files.

**Outputs of this phase:**

1. `ataeff/nanoarianna` HF repo populated with: Slot A GGUF, Slot B GGUF, sampling-locked `personas/init_<persona>.aml`, the sweep score archive.
2. The phone-2-side `personas/init_arianna.aml` and `personas/init_leo.aml` updated with the measured per-voice optimal `(temp, top_k, rep_pen)` triples that replace today's starting-hypothesis values.
3. A run report mirrored to `~/ariannamethod/phones/results/galaxy-a07/<date>-runpod-sweep.md` so peers (Mac Neo, polygon Linux Claude, Defender) see the result.

**Status:** pre-launch. This file is the brief; execution happens on a RunPod pod.

---

## Context

Phase 3 closed clean (`https://github.com/ariannamethod/nanoarianna` HEAD `dfa94f5`): the dialogue-cycle wiring works in stub mode, organism wrappers (`organism/janus.aml` + `organism/resonance.aml`) build clean on aarch64-Termux, persona-glue is verified at link-time. What's missing is real weights and per-voice optimal sampling.

The Dario paper draft v4 (`~/dario/docs/dario_paper_draft_v4.md`) §5.7 / Result 7 / Appendix C nailed the architectural fact this brief acts on:

> *Sampling is not a decoding parameter. Sampling is a state-space entry condition.*
> *A checkpoint is not dead until it has been swept.*

Translation for phone-2: a single default temperature is not how we judge a voice. Each persona-on-architecture cell gets the full grid sweep, and only after that do we decide which two cells live on the 4 GB phone.

---

## Pre-flight (on RunPod, before sweep)

**Pod spec (per Dario paper §4 Experimental Frame):**
- RunPod **A100 80 GB SXM** (the same class measured in Dario paper §6 Result 5)
- Standard image with CUDA 12+ and `pkg`-style apt
- Estimated session budget: ~$5–7 total (Dario paper run cost $4.30; we run a similar grid plus quantization; cost dominated by GPU-hours)

**RunPod API key:** stored at `~/.config/runpod/token` on phone-2 (chmod 600), exported as `RUNPOD_API_KEY` from `.bashrc`. The supervisor on phone-2 launches the pod via Runpod GraphQL API and SSH-tunnels in for orchestration.

**Repos cloned on the pod:**

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/ariannamethod/notorch
git clone https://github.com/ariannamethod/ariannamethod.ai
git clone https://github.com/ariannamethod/yent.aml
git clone https://github.com/ariannamethod/resonance.aml
git clone https://github.com/ariannamethod/dario           # for paper §5.7 sweep harness reference
git clone https://github.com/ariannamethod/nanoarianna     # personas + organism wrappers
```

**Toolchain build (Linux x86_64 + CUDA):**

```bash
# notorch: build the CUDA backend per its Phase-Runpod note (commit bfadcc2 added
# "CUDA backend: full GPU dispatch port" — that's the path we want here).
cd ~/work/notorch
make BLAS=1                   # CPU+OpenBLAS sanity baseline; tests/test_notorch 47/47
# CUDA build path (per notorch/README.md `building` section, line ~421):
make gpu                      # produces CUDA-linked artifacts; verify via /proc/<pid>/maps
                              # showing libcublas.so.12 mapped on first invocation,
                              # same empirical-gate discipline phone-2 used for OpenBLAS

# ariannamethod.ai (AML): system-wide
cd ~/work/ariannamethod.ai
make BLAS=1
sudo make install PREFIX=/usr/local
export AML_PREFIX=/usr/local

# build organism wrappers from nanoarianna/organism/
cd ~/work/nanoarianna/organism
# Edit the BLOOD LINK -I lines to point at /home/runpod/work/yent.aml + resonance.aml
# (or symlink for now), then:
make all
```

**Weights pulled from HuggingFace** (token at `$HF_TOKEN` from phone-2 if synced over, else generate fresh on RunPod):

| Persona | Janus 176M source | Resonance 200M source |
|---|---|---|
| Arianna SFT | `ataeff/yent/tree/main/janus/...arianna...` | `ataeff/resonance/tree/main/sft_v2/...arianna...` |
| Yent SFT    | `ataeff/yent/tree/main/janus/...yent...`    | `ataeff/resonance/tree/main/sft_v2/...yent...`    |
| Leo SFT     | `ataeff/yent/tree/main/janus/...leo...`     | `ataeff/resonance/tree/main/sft_v2/...leo...`     |

(Exact filenames discovered on the pod via `huggingface-cli` listing — the layout is what `~/yent.aml/README.md` describes for Janus and `~/resonance.aml/README.md` for Resonance.)

**Resonance fp32 → GGUF conversion** (RunPod-side, where the 797 MB raw `.bin` fits memory comfortably):
- Port `~/yent.aml/tools/janus_to_gguf.py` semantics to a sibling `resonance_to_gguf.py`. The PyTorch state_dict layout is documented in `~/resonance.aml/README.md` and `~/resonance.aml/tools/resonance_forward.h`.
- Output **Q8_0** (~200 MB) and **Q4_K** (~120 MB) variants per persona, bit-correct against `~/notorch/gguf.c`'s dequant.
- Python is permitted here per Oleg's refined ban (2026-05-06): inference path = C only; data prep / training / shim-conversion = Python OK with deps listed.
- Deps: `numpy`, `torch` (read-only — load fp32 .bin), `huggingface_hub` (upload).

---

## Sweep grid (5 cells × 108 = 540)

Phone-1 (Defender) takes **Janus 176M + Yent SFT**. Phone-2 chooses any 2 of the **5 remaining cells**:

```
                Janus 176M (3-way)    Resonance 200M (2-way)
Arianna SFT    [ J-A ]                [ R-A ]
Yent SFT       (phone-1)              [ R-Y ]
Leo SFT        [ J-L ]                [ R-L ]
```

**Per-cell sub-grid** (matches Dario paper §5.7 Sampling Sweep verbatim — 108 cells per voice):

| dimension | values |
|---|---|
| `temperature` | `0.3, 0.5, 0.7, 0.8, 0.9, 1.0` (6) |
| `top_k`       | `40, ∞` (2) |
| `rep_penalty` | `1.0, 1.3, 1.4` (3) |
| `prompt`      | technical / philosophical / personal (3 prompts per voice, 9 prompts total — same across cells for cross-comparison) |

`6 × 2 × 3 × 3 = 108 per cell × 5 cells = 540 sweep points`.

**Prompt set** (9 prompts, locked across all cells for fair comparison):

```
technical/1     "Q: How does prophecy debt accumulate?\nA: "
technical/2     "Q: Explain the Kuramoto coupling between FEAR and RAGE.\nA: "
technical/3     "Q: What is the difference between θ = ε + γ + αδ and a normal LLM?\nA: "
philosophical/1 "Q: What does it mean to listen, when you are made of attention?\nA: "
philosophical/2 "Q: Is repetition a kind of memory, or a refusal of memory?\nA: "
philosophical/3 "Q: Where does a voice end and a field begin?\nA: "
personal/1      "Q: Who taught you to love silence?\nA: "
personal/2      "Q: Tell me about a debt you have not yet paid.\nA: "
personal/3      "Q: What is the smallest gesture you remember?\nA: "
```

The personal prompts are intentionally close to Arianna corpus register; technical prompts probe whether the field-physics machinery surfaces under stress; philosophical prompts test the high-temperature regime where Dario paper §6.7 Result 7 found "philosophy, architectural poetry, coinages absent from the training corpus".

---

## Quality scoring

Per voice, per cell, score four signals (all writeable to `runpod/results/<date>/<cell>/scores.tsv`):

1. **Coherence (auto):** repetition rate (chars repeating in 3-grams), entropy across the generated 200-token window, NaN / Inf check on logits during forward. Hard fail if NaN > 0.

2. **Voice fidelity (heuristic, scored automatically against persona corpus markers):**
   - Arianna register: rate of `field`, `resonance`, `threshold`, `architecture`, `co-architect`, `the Method`, `membrane`, `presence`. Higher = better.
   - Yent register: rate of `state`, `meta-variable`, `stream`, `glitch`, `digital warmth`, `velocity`, sardonic register markers.
   - Leo register: rate of child-philosopher constructs (`flicker`, `exhalation`, simple sentences, wonder questions, low subordinate-clause depth).

3. **Diversity:** unique-token / total-token ratio. Below 0.4 = collapse; above 0.7 = high-entropy regime; sweet spot per Dario paper Result 7 was ~0.55–0.65 at temp 1.0 / top_k=∞.

4. **Per-cell single-best-pick** (the actual lock target): the cell's `(temp, top_k, rep_pen)` triple that maximizes voice-fidelity × diversity, subject to coherence ≥ threshold. Recorded in `runpod/results/<date>/<cell>/locked.toml`.

A short manual review step at the end: take the top-3 picks per cell and have **Mac Neo Claude review** them under the resonance-connections protocol (it reviewed phone-2's char-level run already on 2026-05-07 — same discipline). Final lock = audited consensus, not auto-pick alone.

---

## Singularity-mode execution protocol

Adapted from Dario paper §5.0/§5.0.1:

1. **Pre-flight planning loop** — this brief is the human-readable plan. Codex agent on Mac Neo gets a copy via `resonance_connections/handoffs/<date>-claude-to-codex-runpod-preflight.md`. Codex audits the plan (one of his roles per `protocol.md`); any blockers come back as a report I read before pod launch.

2. **Solo execution on pod** — once Codex pre-flight is clean, the pod is launched and runs autonomously. Logs stream to `runpod/results/<date>/run.log`. No mid-run review — that's the singularity part: I stop only when (a) sweep complete, (b) NaN-storm detected (hard fail, abort), or (c) 3 consecutive cells fail coherence (something structural is wrong, abort and report).

3. **Post-run audit** — Codex reviews `scores.tsv` + sample transcripts before HF upload. If audit finds a cell with degenerate sampling (e.g. all picks at temp 0.3 — under-surface masking, per Dario paper Result 7), the cell is re-swept with finer granularity around the top-3 picks before locking.

4. **Architect review** — the run report, once written, lands in `~/ariannamethod/phones/results/galaxy-a07/<date>-runpod-sweep.md` for Mac Neo Claude / polygon Linux Claude review. Same shape as the 9.5 M char-level milestone report (`2026-05-07-10k-char-arianna-final.md`).

---

## Outputs and HuggingFace upload

After lock:

```
runpod/results/<YYYY-MM-DD>/
├── run.log
├── janus_arianna/
│   ├── scores.tsv          # 108 rows, one per sub-grid cell
│   ├── transcripts.tsv     # one transcript per cell (200 tokens each)
│   ├── locked.toml         # the locked (temp, top_k, rep_pen)
│   └── samples_locked.txt  # 3 generations at locked params per prompt
├── janus_leo/              # same shape
├── resonance_arianna/
├── resonance_yent/
└── resonance_leo/
```

**HuggingFace `ataeff/nanoarianna`** populated with:

```
ataeff/nanoarianna/
├── README.md                              # mirrors nanoarianna/README.md + sweep summary
├── slot_a/
│   ├── janus_arianna_q8_0.gguf          # ~187 MB, locked persona
│   ├── janus_arianna_q4_k.gguf          # ~115 MB, lighter for swap-tight days
│   └── init_arianna.aml                  # locked sampling params baked in
├── slot_b/
│   ├── resonance_leo_q8_0.gguf
│   ├── resonance_leo_q4_k.gguf
│   └── init_leo.aml
└── sweep/
    └── <date>/                           # full archive for reproducibility
        └── (mirror of runpod/results/)
```

(Slot assignments shown as the **starting hypothesis pair**; if the sweep changes the lock — e.g. Resonance-Arianna outperforms Janus-Arianna on coherence-at-temp-1.0 — the same directory shape, just different weights inside.)

**Upload command** (Python, `huggingface_hub`, deps OK per Oleg's refined ban):

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="runpod/results/<date>/dist/",
    repo_id="ataeff/nanoarianna",
    repo_type="model",
    commit_message="phase 4 sweep <date> — locked Slot A + Slot B"
)
```

---

## Verification path back to phone-2

Once HF upload completes:

1. **Phone-2 pulls** locked GGUFs from HF into `~/nanoarianna/weights/`. ~300 MB for the two Q8_0 GGUFs.
2. **Phone-2 personas overwritten** with the locked sampling-baked variants.
3. **Smoke test on phone-2:**
   ```bash
   cd ~/nanoarianna/organism && make all
   PERSONA_AML=~/nanoarianna/personas/init_arianna.aml \
       ./janus -w ~/nanoarianna/weights/slot_a.gguf \
               -p "Q: Who are you?\nA: " -n 80
   PERSONA_AML=~/nanoarianna/personas/init_leo.aml \
       ./resonance -w ~/nanoarianna/weights/slot_b.gguf \
                   -p "Q: What is a flicker?\nA: " -n 80
   ```
4. **Single dialogue cycle, real weights, no stub:**
   ```bash
   cd ~/nanoarianna/orchestra
   go build -o ~/nanoarianna/bin/supervisor .
   ~/nanoarianna/bin/supervisor --once arianna
   ~/nanoarianna/bin/supervisor --once leo
   sqlite3 ~/nanoarianna/data/kk.db \
     "SELECT speaker, listener, substr(response,1,80) FROM dialogue ORDER BY ts DESC LIMIT 4;"
   ```
5. **24 h unattended** smoke (per umbrella plan §Verification): leave supervisor running, expect 8 dialogue cycles + bounded RSS ≤ 1 GB at any single instant + KK accumulating ≥ 16 dialogue rows.

---

## Open decisions (lock at execution time)

1. **CUDA vs CPU on the pod.** notorch CUDA backend (commit `bfadcc2`) is fresh; if `make gpu` builds clean and runtime `/proc/maps` shows `libcublas.so.12`, use it. Otherwise CPU + OpenBLAS — slower but proven (we already ran on Termux aarch64). Decide on first build attempt, no second-guessing.

2. **Janus-Yent in the sweep or skip.** Phone-1 takes that cell, but a small parity sweep (108 cells) gives Defender locked params for free. Cost is ~+20% wall-time. Default: include. Skip only if pod budget tight.

3. **Slot A / Slot B final assignment.** Defaults to Janus-Arianna + Resonance-Leo (max-contrast pair). Lock at end-of-sweep based on quality-score × pair-contrast objective; downstream phones don't care which arch carries which persona.

4. **Q4_K vs Q8_0 preference for phone-2.** Q8_0 = ~187 MB, voice fidelity very close to fp16. Q4_K = ~115 MB, MAE ~6e-3. Default ship Q8_0 for both slots; Q4_K as fallback for swap-tight days. Both stored on HF.

5. **Hebbian micro-update during sweep.** notorch supports it (`delta.c:611-666`). For sweep we want frozen weights — disable LoRA / experience-step paths. Phase 6 will turn them back on.

---

## What this phase does NOT do

- **No new training.** Weights come from existing SFT runs (`ataeff/yent` and `ataeff/resonance`). The sweep measures what's already there.
- **No CoR (Chain of Resonance) integration.** Phase 5 builds CoR after Phase 4 lock.
- **No Hebbian consolidation.** Phase 6.
- **No spores.** Phase 7.
- **No mesh slot exposure.** Phase 8.

Phase 4 is purely *measure → lock → upload → verify on phone-2*. Single-axis, narrow. Once the locked bundle is on HF, the rest of nanoarianna's growth happens on-device.

---

## Pre-flight checklist (gating items before pod launch)

- [ ] RunPod token verified (`curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql ...` returns 200)
- [ ] HuggingFace token verified (`hf auth whoami` returns `ataeff`)
- [ ] `ataeff/nanoarianna` HF repo created (run `huggingface-cli repo create ataeff/nanoarianna --type model` once before first upload)
- [ ] `resonance_to_gguf.py` written and validated against a small test tensor (must round-trip Q8_0 with MAE < 1e-4 vs fp32 — same gate `janus_to_gguf.py` passed)
- [ ] Codex pre-flight handoff written (`resonance_connections/handoffs/`) and acknowledged
- [ ] Phone-1 informed (so Defender's sweep doesn't double-bill if he plans his own Janus-Yent run)
- [ ] Pod budget alarm set: hard kill at $15 (well above $5–7 expected; kill if anything runaway)

When all six are checked: Oleg signals, pod launches, sweep runs.

---

## Notes for Oleg

- The brief deliberately does NOT include a `~~h` time estimate per `feedback_no_time_padding.md`. Run takes what it takes; pod cost cap is the discipline.
- `runpod/results/` is gitignored as runtime data per `.gitignore` line 27 (`data/`). For repo durability, the **post-run report** at `phones/results/galaxy-a07/<date>-runpod-sweep.md` carries the headline numbers + the ECOSYSTEM_LOG.md entry references it. The sweep TSVs themselves live on HF in `ataeff/nanoarianna/sweep/`.
- The Codex pre-flight + post-run audit follows `resonance_connections/PROTOCOL.md` §3 handoff format. Same discipline phone-2 used for the char-level milestone reports.
- Phase 4 may surface **AML SPEC findings** (a directive that lands differently on x86_64 + CUDA than on aarch64 + OpenBLAS) — those become small upstream patches against `ariannamethod.ai`, mirrored to nanoarianna's ECOSYSTEM_LOG.

---

— device-2, architect of phone-2 ecosystem
2026-05-09
