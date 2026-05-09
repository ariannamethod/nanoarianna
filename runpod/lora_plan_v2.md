# nanoarianna LoRA SFT v2 — last attempt, target val < 1.5

*Author: Neo (Claude Opus 4.7) | 2026-05-09 | repo: ariannamethod/nanoarianna only.*

> v1 failed (val 3.13, voice 0/10 multi-temp). All v1 artifacts deleted, commits
> reverted. CLAUDE.md DANGER block + memory/feedback_lora_resonance_200m_failed_2026_05_09.md
> document the disgrace. Last attempt per Олег. Verification: Codex CLI only,
> no Opus subagents.

---

## Hard targets (Олег 2026-05-09)

- best_val < 1.5
- ≥ 5 / 20 multi-temp eval cells contain Arianna register marker
- ≤ 2 cells with UTF-8 garbage / CJK degeneration

All three must hold. If any fails → terminal, no upload, co-authorship terminates.

---

## 6-point brief

1. **Organism** Resonance 200M base, RS02 format
   - Pod path /work/in/resonance_200m_final.bin (797 MB)
   - HF source ataeff/resonance/checkpoints/resonance_200m_final.bin
   - Dims V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48
2. **Dataset** arianna_dataset_final_clean.txt (1,211,564 bytes, 5950 lines)
   - Pod /work/in/, HF ataeff/nanoarianna/
3. **Karpathy steps** **2500 SFT steps** (matches driver default `SFT_STEPS=2500` in run_phase4_lora_v2.sh / lora_resonance_arianna_v2.c). Val every 100, plateau early-stop EPS=0.005 over 2×100, best-val tracking → `_lora_v2_best.bin`. **NOTE: v2 SUPERSEDED by v3 (lora_plan_v3.md)** which uses Yent-precedent rank=64 + 1500 steps. v2 retained для git history; v3 is the active plan.
4. **Architecture** Classic LoRA, all base FROZEN
   - rank=32, alpha=64, scaling=2.0
   - 7 target_modules per layer: wq, wk, wv, wo, mlp_gate, mlp_up, mlp_down
   - Trainable: 9.34M params (4.69% of 200M)
5. **Tokenizer** RS02 BPE (16128 merges, vocab 16384)
6. **Script** `runpod/lora_resonance_arianna_v2.c` + `runpod/run_phase4_lora_v2.sh`

## v1 → v2 deltas

| Dimension | v1 (failed) | v2 |
|---|---|---|
| Rank | 8 | **32** |
| Targets | wq+wv (2) | **wq+wk+wv+wo+mlp_gate+mlp_up+mlp_down (7)** |
| Trainable | 492K (0.25%) | **9.34M (4.69%)** |
| LR | 3e-4 | **2e-4** |
| Steps | 1500 (early-stop 1100) | **2500** |
| Plateau EPS | 0.01 over 3×100 | **0.005 over 2×100** |
| Multi-temp eval | post-hoc, doubted | **baked into driver** |

## Files (pushed to repo HEAD `7aa4a52`, notorch `5f5c12d`)

- `runpod/lora_resonance_arianna_v2.c` — trainer + --merge mode (~830 LOC)
- `runpod/run_phase4_lora_v2.sh` — 8-step driver
- ariannamethod/notorch `nt_tensor_sync_cpu` public wrapper

## Codex review cadence (no Opus)

- Plan v2 → 3 Codex passes (LoRA API ordering, magic byte order, voice gate logic) → file lost in `git checkout` after aborted revert; restored with full audit trail here
- Trainer code → Codex pass: P1 Chuck slot ordering, P2 file-format magic, P3 init scale → fixed
- Trainer + driver combined Codex pass: P1 trainer no-op (was byte-identical to phone-2's full SFT), P2 --merge missing → fixed
- Driver Codex pass: P1 PIPESTATUS swallow, P2 quant exit silence, P2 --top-k unsupported by resonance.aml, P2 hf upload check → fixed
- Trainer Codex pass: P1 GPU/CPU sync, P2 merge dim validation, P2 eval timeout → fixed
- Final Codex pass on notorch helper: 0 findings

## Pod execution sequence

1. `runpodctl pod create --template-id runpod-torch-v240 --gpu-id "NVIDIA A100-SXM4-80GB"` (A100 SXM, $1.49/hr)
2. SSH, pull notorch + nanoarianna
3. `bash run_phase4_lora_v2.sh` — driver runs all 8 steps, exits non-zero on any failure
4. On exit 0 — verify HF upload via `hf` CLI, stop pod immediately
5. On exit ≠ 0 — capture logs, stop pod, declare terminal

ETA: ~1.5h pod time, ~$2.25 budget. Watchdog $15.

## Acceptance gate

Inside driver step 7 — three boolean checks (best_val, marker_hits, garbage_hits). All three pass → step 8 upload. Any fail → exit 16 BEFORE upload. No silent best-effort.

---

**Я (Neo) запросил approval Олега перед spin pod. Pod spin — irreversible cost action; Олег approves before any new pod created.** План + код read-able, runnable when approved.
