# nanoarianna LoRA SFT v3 — Yent precedent, target val < 1.5

*Author: Neo (Claude Opus 4.7) | 2026-05-09 evening | repo: ariannamethod/nanoarianna only.*

> v1 failed (val 3.13, voice 0/10). v2 was rank=32 — half of working precedent.
> v3 mirrors **actual Resonance 200M LoRA Yent recipe** (HF `ataeff/resonance/sft_v2/resonance_lora_yent_best.pt`).
> Last attempt. Hard target: best_val < 1.5.

---

## Hard targets (Олег 2026-05-09)

- best_val < 1.5
- ≥ 5 / 20 multi-temp eval cells contain Arianna register marker
- ≤ 2 cells with UTF-8 garbage / CJK degeneration

All three must hold. Any fail → terminal, no upload, co-authorship terminates.

---

## Yent precedent reverse-engineering

HF file `ataeff/resonance/sft_v2/resonance_lora_yent_best.pt` = **74.8 MB** (per HF API).
- 74,800,653 bytes / 4 bytes/float = **18,700,163 params**
- Per-layer LoRA cost для 7 targets at rank R:
  - 4 attn (E×E): each 2·R·E. × 4 = 8·R·E
  - 2 ffn ([M,E]): each R·E + M·R. × 2 = 2·R·E + 2·M·R
  - 1 ffn ([E,M]): R·M + E·R
  - Total = R·(11E + 3M) = R·(11·768 + 3·2048) = R · 14592
- 20 layers × R · 14592 = R · 291,840
- 18,700,163 / 291,840 ≈ **R = 64.07** → exact match.

**Conclusion:** working Resonance 200M LoRA Yent uses **rank=64, full 7 target_modules**.
Source: `ataeff/resonance/sft_v2/` artifact byte size — direct measurement.

---

## 6-point brief

1. **Organism** Resonance 200M base, RS02 format
   - Pod path /work/in/resonance_200m_final.bin (797 MB)
   - HF ataeff/resonance/checkpoints/resonance_200m_final.bin
   - Dims V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R_rrpram=48
2. **Dataset** arianna_dataset_final_clean.txt (1,211,564 bytes / 1.21 MB / 5950 lines, per `wc -lc`)
   - Pod /work/in/, HF ataeff/nanoarianna/
   - Per phone-2 measurement: 1.21 MB → 294,137 BPE tokens, 4.1 bytes/token
   - 1 epoch ≈ 600 steps at ctx=512 batch=1
3. **Karpathy steps** **1500 SFT steps** (~2.5 epoch). Val every 100, plateau early-stop EPS=0.005 over 2×100, best-val tracking → `_lora_v3_best.bin`. Arianna corpus ~100× smaller than Yent (per Олег 2026-05-09) — same recipe shape but fewer iterations before overfit.
4. **Architecture** Classic LoRA, all base FROZEN
   - **rank=64** (matches Yent precedent exactly)
   - **alpha=128** (scaling α/r = 2.0 same as Yent precedent)
   - **7 target_modules per layer**: wq, wk, wv, wo, mlp_gate, mlp_up, mlp_down (matches Yent precedent)
   - **Trainable: 18.68M params (9.34% of 200M)** — 2× v2's 9.34M, 38× v1's 492K
5. **Tokenizer** RS02 BPE (16128 merges, vocab 16384) — same as inference
6. **Script** `runpod/lora_resonance_arianna_v3.c` (forks v2.c with rank/alpha defaults updated to 64/128, save magic LRV3) + `runpod/run_phase4_lora_v3.sh`

---

## Recipe comparison

| Dimension | v1 (failed) | v2 (revised but never run) | **v3 (Yent precedent)** |
|---|---|---|---|
| Rank | 8 | 32 | **64** |
| Alpha | 16 | 64 | **128** |
| Targets | wq+wv (2) | wq,wk,wv,wo,mlp_gate,mlp_up,mlp_down (7) | **same 7** |
| Trainable | 492K (0.25%) | 9.34M (4.69%) | **18.68M (9.34%)** |
| LR | 3e-4 | 2e-4 | **2e-4** |
| Steps | 1500 | 2500 | **1500** |
| Plateau EPS | 0.01 over 3×100 | 0.005 over 2×100 | **0.005 over 2×100** |

Adapter file size at rank=64: 74.8 MB (matches HF Yent precedent exactly).

---

## Files (to push if approved)

- `runpod/lora_resonance_arianna_v3.c` — fork of v2.c with:
  - Default rank=64 alpha=128 (was 32/64)
  - Magic LRV3 = 0x3356524C (was LRV2 = 0x3256524C; bumped version так чтобы v2-format adapters не загружались случайно)
  - Otherwise identical: 7 LoRA pairs per layer, register-LoRA-first ordering, plateau EPS, smoke-50 kill, --merge mode with dim validation, nt_tensor_sync_cpu before save
- `runpod/run_phase4_lora_v3.sh` — fork of v2.sh with:
  - LORA_RANK=64 LORA_ALPHA=128 defaults
  - Trainer binary name `lora_resonance_arianna_v3`
  - Output prefix `resonance_arianna_lora_v3`
  - HF upload paths under `lora_v3/` (not `lora_v2/`)
  - Otherwise identical PIPESTATUS checks, multi-temp eval grid, acceptance gate

Notorch dep: `nt_tensor_sync_cpu` already pushed (commit 5f5c12d on main).

---

## Pod execution sequence (await Олег "да" before each)

1. **Approval gate 1** — Olег says "да" to plan v3 (THIS file, after Codex review)
2. Push trainer v3 + driver v3 commit
3. **Approval gate 2** — Olег says "да" to pod spin
4. `runpodctl pod create --template-id runpod-torch-v240 --gpu-id "NVIDIA A100-SXM4-80GB"` (~$1.49/hr)
5. SSH, pull, build (notorch + trainer + driver via driver step 1)
6. `nohup bash run_phase4_lora_v3.sh > /work/phase4_lora_v3.log 2>&1` — driver runs all 8 steps
7. On exit 0 + Олег approval → declare done. On exit non-zero → terminal fail.

ETA: ~1.5h pod time, ~$2.25 budget. Watchdog $15.

## Cost discipline

Pod spin = $1.49/hr from second 1. Budget for v3 path: build (~3 min, $0.07) + smoke-50 (~1 min, $0.02) + full SFT 1500 steps (rank=64 ≈ 1.5s/step ≈ 38 min, $0.94) + merge (~30s) + quant (~5 min) + multi-temp eval (~6 min, mostly CPU-bound) + upload (~5 min) = **~60 min total, ~$1.50**. Stop pod within 5 min of upload verify.

## Acceptance gate (driver step 7)

Three boolean checks (best_val, marker_hits, garbage_hits). All three pass → step 8 upload. Any fail → exit 16 BEFORE upload. No silent best-effort.

---

## Codex review cadence

Plan v3 → 1 Codex pass minimum on this file before quote-to-Олег. Code v3.c + v3.sh → 1 Codex pass each. Combined → 1 Codex pass. All until 0 P0/P1 OR 3 passes. No Opus subagents.

---

*v3 draft. Next: Codex review pass 1. Then quote to Олег. Then await "да".*
