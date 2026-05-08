# ECOSYSTEM_LOG

Chronological journal of nanoarianna decisions and events. Newest at top. Each entry: date, what happened, why, what's next. Don't rewrite history; correct via new entry.

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
