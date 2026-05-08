# ECOSYSTEM_LOG

Chronological journal of nanoarianna decisions and events. Newest at top. Each entry: date, what happened, why, what's next. Don't rewrite history; correct via new entry.

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
