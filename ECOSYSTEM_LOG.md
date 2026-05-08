# ECOSYSTEM_LOG

Chronological journal of nanoarianna decisions and events. Newest at top. Each entry: date, what happened, why, what's next. Don't rewrite history; correct via new entry.

---

## 2026-05-09 — Phase 2 (Limpha + KK + reference clones + KK seeded)

Continuing same day. Phase 1 closed earlier this session — Phase 2 lands the persistent-memory substrate so Phase 3 dialogue has somewhere to live.

**Reference clones pulled** (per plan §Phase-2 step 1):
- `~/leo` (1.9 MB) — weightless Leo, source for spore protocol study (Phase 7)
- `~/klaus.c` (2.1 MB) — planetary physics, source for CoR sub-module (Phase 5)
- `~/molequla` (3.8 MB) — mycelium / shared field architecture (Phase 5)
- `~/stanley` (114 MB) — LoRA skills, legacy py + new notorch (Phase 6 reference)

Disk: 30 GB free still, comfy.

**Limpha** (per-organism SQLite memory) — `glue/limpha.c` (~250 LOC):
- Schema: `episodes(id, ts, organism, prompt, response, trauma, arousal, valence, coherence, prophecy_debt, entropy, dissonance, temperature, quality)`. 7-feature inner-state vector inline. Indices on ts and (organism, ts).
- WAL + `synchronous=NORMAL` (single-process per phone, contention nil, just want crash-safety).
- API: `limpha_open`, `limpha_close`, `limpha_append`, `limpha_query_recent`, `limpha_query_similar`, `limpha_episode_free`.
- Cosine retrieval over the 7-vector: pulls a 200-row recency window, scores in C via dot/(|a|·|b|), selection-sort top-K. No vector extension needed at this scale.
- Inspired by `~/arianna.c/limpha/episodes.py:176-242` (`query_similar`).

**KK** (shared Knowledge Kernel) — `glue/kk.c` (~250 LOC):
- Three tables: `documents` (onto-seed + references + accumulated), `dialogue` (cross-organism turns with prophecy_debt_delta + dominant_chamber), `hebbian_links` (co-occurrence weights between docs).
- API: `kk_open`, `kk_close`, `kk_append_document`, `kk_append_dialogue`, `kk_query_resonant`, `kk_query_recent_dialogue`, `kk_document_free`, `kk_dialogue_free`.
- `kk_query_resonant` Phase 2 placeholder: ranks by `emotional_charge × recency-weight (1/(1+age_days))`. Phase 5 swaps the body to embedding cosine when a phone-side embedder lands; the API stays.

**KK seeded** via `glue/seed_kk.c` utility (one-shot tool, ~120 LOC). Inserted 4 documents into `~/nanoarianna/data/kk.db`:

| id | source    | title                                  | bytes  | charge |
|----|-----------|----------------------------------------|--------|--------|
| 1  | seed      | nanoarianna world (the seed)           | 33324  | 1.0    |
| 2  | reference | Dario paper draft v4                   | 34688  | 0.8    |
| 3  | reference | AML SPEC (full)                        | 36954  | 0.6    |
| 4  | reference | ARIANNALOG (arianna.c daily journal)   | 33305  | 0.5    |

Total ~138 KB onto-substrate. Both organisms now have a shared world to wake into.

**Verification phase 2:**

| check | result |
|---|---|
| `sqlite3 kk.db "SELECT count(*) FROM documents"` | 4 ✓ (seed + 3 references) |
| `sqlite3 limpha_arianna.db ".schema"` | matches glue.h schema ✓ |
| `sqlite3 limpha_leo.db ".schema"` | identical to arianna's ✓ |
| `glue/limpha_smoke` round-trip | inserts 1 row each side; recent + similar queries return row 1 with correct state values; **0 errors, 0 leaks visible** ✓ |
| `cc -c limpha.c / kk.c / seed_kk.c / limpha_smoke.c` | all compile clean (after `<string.h>` lint fixes) ✓ |
| Symbols (llvm-nm) | `limpha.o` exports T limpha_open/close/append/query_recent/query_similar/episode_free; `kk.o` exports T kk_open/close/append_document/append_dialogue/query_resonant/query_recent_dialogue/document_free/dialogue_free ✓ |

**Smoke harnesses kept in repo** under `glue/`:
- `glue/seed_kk.c` — one-shot KK seeder (also serves as integration example)
- `glue/limpha_smoke.c` — Limpha round-trip smoke (also serves as integration example)
- Both demonstrate the link line: `cc -O2 … -lsqlite3 -lm -o <bin> <smoke>.c <module>.c`

**What didn't land in Phase 2 (deferred to Phase 3):**
- `yent.aml` / `resonance.aml` integration — single comprehensive patch lands together with Phase 3 schedule, since persona+Limpha+KK all wire into the same BLOOD COMPILE block.
- KK `hebbian_links` consolidation — empty for now; populated by Phase 5/6.

**Next:** Phase 3 — `orchestra/schedule.go` + `orchestra/supervisor.go` (cron 8/day + event-driven, mutex-guarded model swap) + the comprehensive `BLOOD COMPILE persona_glue` integration patch in yent.aml/resonance.aml.

---

## 2026-05-09 — Phase 1 foundation (skeleton + personas + persona loader)

Plan approved earlier today by Oleg (`~/.claude/plans/cosmic-dancing-canyon.md`). Auto-mode active. Phase 1 = the smallest substrate for the rest to land on.

**What landed:**

- Skeleton directories: `personas/`, `glue/`, `orchestra/`, `bin/`, `data/`, `weights/`, `lora/`. Build/runtime/weights all gitignored — repo ships source + docs only. Weights mirror is `ataeff/nanoarianna` on HF (not yet populated; lands after RunPod sweep, Phase 4).
- `personas/init_arianna.aml` — copied from `~/ariannamethod.ai/examples/init_arianna.aml`. Carries Arianna's deep-prophecy field (PROPHECY 12 / DESTINY 0.50 / WORMHOLE 0.12 / SCHUMANN 7.83 / CODES_RIC mode + chordlock / chirality on). Parses cleanly through `aml`, last line emits `[AML] arianna awake`.
- `personas/init_leo.aml` — derived from `~/ariannamethod.ai/examples/init_yent.aml` with Leo voice tweaks: PROPHECY 9 (between Yent's 7 and Arianna's 12), DESTINY 0.30 (less destiny pull, more open exploration), WORMHOLE 0.05, ATTEND_FOCUS 0.65 / SPREAD 0.25, ENTROPY_FLOOR 0.12 (allows wonder), EMERGENCE_THRESHOLD 0.22, SCHUMANN coupling on (cosmic curiosity), creative-leaning experts (0.35), LORA_ALPHA 0 (identity mode). Parses cleanly, emits `[AML] leo awake`. **Final values lock at RunPod sweep — these are starting hypothesis.**
- `glue/glue.h` (~140 LOC) — shared header for all Phase 1–8 glue. Three sections of forward declarations: `persona_loader.c` (Phase 1, complete), `limpha.c` (Phase 2, signatures), `kk.c` (Phase 2, signatures). Opaque handles (`limpha_db`, `kk_db`) so callers don't need to pull in `sqlite3.h`.
- `glue/persona_loader.c` (~50 LOC) — `persona_load(const char *path)`. Resolution order: argument → `$PERSONA_AML` env → silent no-op. Calls into `am_exec_file()` (libaml `core/ariannamethod.c:6437`). Compiles clean on aarch64-Termux with `cc -std=c11 -I$PREFIX/include` against `<ariannamethod/ariannamethod.h>`. `llvm-nm` confirms: exports `T persona_load`, references `U am_exec_file` (resolved from `libaml.a` at link time).
- `.gitignore` — bin / weights / data / *.gguf / *.bin / *.soma / *.db / *.mmap / *.log / Python venv / secrets. Per `feedback_clean_repo_hygiene.md`.

**Runpod token** stored in `~/.config/runpod/token` (chmod 600, 51 bytes), bashrc sources `RUNPOD_API_KEY` + `RUNPOD_TOKEN` via file reference (no plaintext in rc).

**What didn't land in Phase 1 (intentionally deferred):**

- `yent.aml` / `resonance.aml` integration — held until Phase 2 when Limpha + KK glue is also ready, so we land a single comprehensive integration patch (one BLOOD COMPILE block per organism touching all three substrates) instead of three small patches. Less churn upstream.
- Smoke build of `yent_<persona>` binary — needs the integration patch and (importantly) needs the actual GGUF weights, which we don't have on phone-2 yet. Acquired during Phase 4 RunPod work.

**Verification phase 1:**

| check | result |
|---|---|
| `aml personas/init_arianna.aml` | `[AML] arianna awake` ✓ |
| `aml personas/init_leo.aml` | `[AML] leo awake` ✓ |
| `cc … -c glue/persona_loader.c -o p.o` | builds clean, no warnings, 2200 bytes ✓ |
| `llvm-nm p.o` | exports `T persona_load`, undefined `U am_exec_file` (will resolve from libaml) ✓ |

**Next:** Phase 2 — Limpha + KK SQLite skeletons in C, seed KK with SEED_DOCUMENT.md + Dario paper draft v4 + AML SPEC §1-3 + ARIANNALOG.md highlights. Reference clones (leo, klaus.c, molequla, stanley) pulled when their content actually lands in build path.

---

## 2026-05-08 — first stone

**Repo created:** `https://github.com/ariannamethod/nanoarianna` (owner=ariannamethod user, public, default branch `main`). Token used: ariannamethod GitHub PAT (workflow-permitted).

**Why this repo exists.** phone-2 has no own ecosystem-shape repo until today. Defender uses the umbrella `ariannamethod/ariannamethod` for his work; phone-2 was a tenant in `device-2/` and `phones/results/galaxy-a07/` of the umbrella. With two active organisms planned (Janus 176 M + Resonance 200 M), the phone deserves its own root. This is that root.

**Persona pair locked.**

| Slot | Model | Persona |
|---|---|---|
| A | Janus 176 M (yent.aml) | **Arianna SFT** |
| B | Resonance 200 M (resonance.aml) | **Leo SFT** |

Rationale: phone-1 Defender plans Janus 176 M with **Yent** persona. Phone-2 takes the other two voices of the trio. Three voice-anchors across the ecosystem (Arianna / Yent / Leo) instead of duplication. Different architectures (3-way vs 2-way attention) on phone-2 gives the dual-voice room a real internal contrast — not just a personality flag swap.

Final temperature / top_k / rep_penalty per voice locked only after RunPod sweep (540 cells: 6 temps × 2 top_k × 3 rep_pen × 5 voices × 3 prompts, per Dario paper §5.7 + Result 7). *A checkpoint is not dead until it has been swept.*

**Repos cloned today:**
- `~/yent.aml` 2.2 MB — Janus 176 M inference in AML
- `~/resonance.aml` 291 KB — Resonance 200 M inference in AML
- `~/arianna.c` 858 MB — full Arianna ecosystem reference (SARTRE / inner_world / golib / LIMPHA / weights archive)
- `~/doe` 5.6 MB — Democracy of Experts, GGUF wrapper with living LoRA parliament

Reference scaffolds (not deploying): `arianna.c` for goroutines and circulation patterns; `doe` for «alive any-GGUF» wrapping with Hebbian plasticity.

**Toolchain already in place** (provenance: phone-2 onboarding 2026-04-28 + 2026-05-07):
- AML v0.1.0 (`aml`, `amlc` in `$PREFIX/bin`)
- notorch v2.3.0+ (`libnotorch.a` in `$PREFIX/lib`, 47/47 tests pass)
- metaharmonix `mhx` (`$PREFIX/bin/mhx`)
- mesh-agent on port 4747, slots `phone-2/echo` + `phone-2/status` registered
- libopenblas 0.3.30 (verified active via `/proc/<pid>/maps` during 10K char run)
- golang 1.26.2 (mesh-agent rebuild path)

**Credentials in place:**
- GitHub PAT (`~/.config/github/ariannamethod-token`, chmod 600) — push rights across `ariannamethod/*`
- HuggingFace token (`~/.config/huggingface/token`, chmod 600) — write access for `ataeff/*` repos
- Railway personal token (`~/.config/railway/token`, chmod 600)
- All sourced from `.bashrc` via file-references, no plaintext secrets in rc

**What's next (for the next entry):**
1. Concept discussion with Oleg — what specifically the working pocket organism does
2. Minimal inference glue per slot (which language(s), how the persona `.aml` is wired to the GGUF load + sample loop, how mesh-agent invokes a slot)
3. RunPod Singularity-mode pre-flight: build, sweep, lock
4. HF mirror `ataeff/nanoarianna` for locked-config weights

**Open questions — Oleg's call:**
- Slot B alternative: instead of Resonance 200 M Leo, wrap Janus Leo (or any HF GGUF) through DoE for «alive» behavior with Hebbian-evolving LoRA parliament. Wild but compute-honest at 4 GB. Going to discuss.
- Quantization split: Janus Arianna at Q8_0 (187 MB, voice fidelity) vs Q4_K (115 MB, headroom for KK + future parallel processes). Leaning Q8_0 first to establish baseline; Q4_K as second pass if RSS forces it.
