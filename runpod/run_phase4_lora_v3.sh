#!/bin/bash
# run_phase4_lora_v3.sh — last-attempt LoRA Resonance Arianna SFT v3 driver.
#
# Plan: runpod/lora_plan_v3.md (Yent precedent rank=64, Codex review only, no Opus).
# Recipe: rank=64 alpha=128 lr=2e-4 ctx=512 steps=1500, 7 target modules per layer (matches HF resonance_lora_yent_best.pt 74.8 MB precedent).
# Hard target: best_val < 1.5. Voice gate: ≥5/20 multi-temp cells with marker.
# After v1 failed (val 3.13, voice 0/10) — see CLAUDE.md DANGER block.

set -u
set -o pipefail

WORK="${WORK:-/work}"
IN="${IN:-$WORK/in}"
OUT="${OUT:-$WORK/out}"
NANOREPO="${NANOREPO:-$WORK/nanoarianna}"
NOTORCH="${NOTORCH:-$WORK/notorch}"
AML="${AML:-$WORK/ariannamethod.ai}"
HF_REPO="${HF_REPO:-ataeff/nanoarianna}"
BUDGET_USD="${BUDGET_USD:-15}"
SFT_STEPS="${SFT_STEPS:-1500}"
SFT_LR="${SFT_LR:-2e-4}"
SFT_CTX="${SFT_CTX:-512}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

DATE_TAG="$(date +%F)"
TIMESTAMP="$(date +%FT%H%M%S)"
RUN_REPORT="$OUT/run_report_lora_v3_${TIMESTAMP}.txt"

mkdir -p "$OUT" "$OUT/sft_v2" "$OUT/slot_arianna"

log() { echo "[lora_v3 $(date +%T)] $*" | tee -a "$RUN_REPORT"; }

log "════════════════════════════════════════════════════════"
log "  LoRA SFT v3 — Resonance 200M + Arianna corpus"
log "  rank=$LORA_RANK alpha=$LORA_ALPHA lr=$SFT_LR steps=$SFT_STEPS ctx=$SFT_CTX"
log "  budget=\$$BUDGET_USD  HF=$HF_REPO"
log "════════════════════════════════════════════════════════"

# ─── 0. Pre-flight: HF download base + corpus ────────────────────────────
log "─── step 0: HF pre-flight download ───"
mkdir -p "$IN"
if [ -z "${HF_TOKEN:-}" ]; then
    log "FATAL: HF_TOKEN unset"; exit 9
fi
hf_pull() {
    local repo="$1" path="$2" dst="$3"
    if [ -f "$dst" ]; then log "  cached: $dst"; return 0; fi
    log "  fetching $repo/$path → $dst"
    curl -fL --retry 3 -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/$repo/resolve/main/$path" -o "$dst" 2>&1 | tail -2
}
hf_pull "ataeff/resonance"   "checkpoints/resonance_200m_final.bin" "$IN/resonance_200m_final.bin"
hf_pull "ataeff/nanoarianna" "arianna_dataset_final_clean.txt"      "$IN/arianna_dataset_final_clean.txt"
[ -s "$IN/resonance_200m_final.bin" ] || { log "FATAL: base bin missing"; exit 9; }
[ -s "$IN/arianna_dataset_final_clean.txt" ] || { log "FATAL: corpus missing"; exit 9; }
log "step 0 OK"

# ─── 1. Toolchain build (notorch CPU + GPU + AML + LoRA trainer) ─────────
log "─── step 1: toolchain + LoRA trainer build ───"
{
    cd "$NOTORCH"
    export PATH="/usr/local/cuda/bin:$PATH"
    log "  notorch GPU lib..."
    make clean >/dev/null 2>&1
    if ! make USE_CUDA=1 lib -j8 2>&1 | tail -3; then log "FATAL: notorch GPU build"; exit 10; fi
    if ! make USE_CUDA=1 install PREFIX=/usr/local 2>&1 | tail -3; then log "FATAL: notorch install"; exit 10; fi
    log "  notorch CPU lib..."
    make clean >/dev/null 2>&1
    if ! make lib -j8 2>&1 | tail -3; then log "FATAL: notorch CPU build"; exit 10; fi
    if ! make install PREFIX=/usr/local 2>&1 | tail -3; then log "FATAL: notorch CPU install"; exit 10; fi

    cd "$AML"
    log "  ariannamethod (libaml + amlc)..."
    make clean >/dev/null 2>&1
    if ! make -j8 2>&1 | tail -3; then log "FATAL: AML build"; exit 10; fi
    if ! make install PREFIX=/usr/local 2>&1 | tail -3; then log "FATAL: AML install"; exit 10; fi

    log "  LoRA SFT v3 trainer build (USE_CUDA)..."
    cd "$NANOREPO/runpod"
    if ! cc -O3 -Wall -DUSE_BLAS -DUSE_CUDA lora_resonance_arianna_v3.c \
            -I/usr/local/include \
            -L/usr/local/lib -L/usr/local/cuda/lib64 \
            -lnotorch_gpu -laml \
            -lopenblas -lcudart -lcublas -lm -lpthread \
            -o lora_resonance_arianna_v3 2>&1 | tail -10; then
        log "FATAL: LoRA trainer build"; exit 10
    fi
} || exit 10
log "step 1 OK — binaries built"

# ─── 2. Verify ────────────────────────────────────────────────────────────
log "─── step 2: verify ───"
[ -f /usr/local/lib/libnotorch.a ]      || { log "FATAL: libnotorch.a missing"; exit 11; }
[ -f /usr/local/lib/libnotorch_gpu.a ]  || { log "FATAL: libnotorch_gpu.a missing"; exit 11; }
[ -x "$NANOREPO/runpod/lora_resonance_arianna_v3" ] || { log "FATAL: trainer binary"; exit 11; }
log "step 2 OK"

# ─── 3. Smoke-50 ─────────────────────────────────────────────────────────
SMOKE_PREFIX="$OUT/sft_v2/lora_v3_smoke50"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    log "─── step 3: smoke-50 SKIPPED ───"
else
    log "─── step 3: smoke-50 LoRA SFT (50 steps, kill on bad signal) ───"
    "$NANOREPO/runpod/lora_resonance_arianna_v3" \
        "$IN/resonance_200m_final.bin" \
        "$IN/arianna_dataset_final_clean.txt" \
        "$SMOKE_PREFIX" \
        50 "$SFT_LR" "$SFT_CTX" "$LORA_RANK" "$LORA_ALPHA" 2>&1 | tee -a "$RUN_REPORT"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then log "FATAL: smoke-50 exited $rc"; exit "$rc"; fi
    log "step 3 OK — smoke-50 PASS"
    rm -f "${SMOKE_PREFIX}_lora_v3_best.bin" "${SMOKE_PREFIX}_lora_v3_final.bin"
fi

# ─── 4. Full LoRA SFT ────────────────────────────────────────────────────
SFT_PREFIX="$OUT/sft_v2/resonance_arianna_lora_v3"
log "─── step 4: full LoRA SFT v3 ($SFT_STEPS steps) ───"
"$NANOREPO/runpod/lora_resonance_arianna_v3" \
    "$IN/resonance_200m_final.bin" \
    "$IN/arianna_dataset_final_clean.txt" \
    "$SFT_PREFIX" \
    "$SFT_STEPS" "$SFT_LR" "$SFT_CTX" "$LORA_RANK" "$LORA_ALPHA" 2>&1 | tee -a "$RUN_REPORT"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then log "FATAL: full SFT exited $rc"; exit "$rc"; fi
LORA_BEST="${SFT_PREFIX}_lora_v3_best.bin"
LORA_FINAL="${SFT_PREFIX}_lora_v3_final.bin"
if [ -f "$LORA_BEST" ]; then
    LORA_USE="$LORA_BEST"
    log "step 4 OK — adapter (best-val): $LORA_BEST"
else
    LORA_USE="$LORA_FINAL"
    log "step 4 OK — adapter (final, no best): $LORA_FINAL"
fi

# Extract best_val from log for hard-target gate later.
BEST_VAL=$(grep "best_val=" "$RUN_REPORT" | tail -1 | sed -E 's/.*best_val=([0-9.]+).*/\1/')
log "  best_val measured: $BEST_VAL"

# ─── 4.5. Merge LoRA into base RS02 ──────────────────────────────────────
MERGED_BIN="$OUT/sft_v2/resonance_arianna_lora_v3_merged.bin"
log "─── step 4.5: merge LoRA → $MERGED_BIN ───"
"$NANOREPO/runpod/lora_resonance_arianna_v3" --merge \
    "$IN/resonance_200m_final.bin" \
    "$LORA_USE" \
    "$MERGED_BIN" 2>&1 | tee -a "$RUN_REPORT"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then log "FATAL: merge exited $rc"; exit "$rc"; fi
[ -s "$MERGED_BIN" ] || { log "FATAL: merged bin empty"; exit 13; }
log "step 4.5 OK"

# ─── 5. Quantize merged → GGUF Q8_0 + Q4_K ───────────────────────────────
log "─── step 5: quantize merged base → GGUFs ───"
log "  Q8_0..."
python3 "$NANOREPO/runpod/resonance_to_gguf.py" \
    "$MERGED_BIN" \
    "$OUT/slot_arianna/resonance_arianna_lora_v3_q8_0.gguf" \
    --quant q8_0 2>&1 | tail -5
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then log "FATAL: Q8_0 quant exit $rc"; exit 14; fi
[ -s "$OUT/slot_arianna/resonance_arianna_lora_v3_q8_0.gguf" ] || { log "FATAL: Q8_0 gguf empty"; exit 14; }
log "  Q4_K..."
python3 "$NANOREPO/runpod/resonance_to_gguf.py" \
    "$MERGED_BIN" \
    "$OUT/slot_arianna/resonance_arianna_lora_v3_q4_k.gguf" \
    --quant q4_k 2>&1 | tail -5
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then log "FATAL: Q4_K quant exit $rc"; exit 14; fi
[ -s "$OUT/slot_arianna/resonance_arianna_lora_v3_q4_k.gguf" ] || { log "FATAL: Q4_K gguf empty"; exit 14; }
log "step 5 OK"

# ─── 6. Multi-temp inference smoke (Phase 7 discipline) ──────────────────
log "─── step 6: multi-temp eval (5 temps × 2 top_p × 2 prompts = 20 cells) ───"
RES_BIN="$NANOREPO/organism/resonance"
if [ ! -x "$RES_BIN" ]; then
    log "  building resonance organism via amlc..."
    cd "$NANOREPO/organism" && make resonance 2>&1 | tail -3
    [ -x "$RES_BIN" ] || { log "FATAL: resonance build"; exit 15; }
    cd - >/dev/null
fi
EVAL_LOG="$OUT/sft_v2/lora_v3_eval.txt"
: > "$EVAL_LOG"
PROMPTS=("Я Арианна, и я" "Резонанс не сломан. Я")
TEMPS=(0.3 0.5 0.7 0.9 1.0)
# resonance.aml CLI parser supports --top-p but NOT --top-k (per Codex Pass 5 P2).
# Vary top-p sampling to get distinct cells.
TOPPS=(0.85 0.95)
MARKER_PATTERNS="резонанс|поле|Арианна|method|Метод|recursion|emergence|architect|архитектор"
MARKER_HITS=0
GARBAGE_HITS=0
TOTAL=0
for PROMPT in "${PROMPTS[@]}"; do
  for T in "${TEMPS[@]}"; do
    for P in "${TOPPS[@]}"; do
      TOTAL=$((TOTAL + 1))
      LABEL="t${T}_p${P}_$(echo "$PROMPT" | tr ' ,.' '___' | head -c 24)"
      OUT_TXT="$OUT/sft_v2/eval_${LABEL}.txt"
      timeout 60 "$RES_BIN" -w "$MERGED_BIN" -p "$PROMPT" \
          -n 80 -t "$T" --top-p "$P" --rep-pen 1.3 > "$OUT_TXT.raw" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
          log "  cell $LABEL: exit $rc — eval failure (timeout or crash)"
          GARBAGE_HITS=$((GARBAGE_HITS + 1))
      fi
      sed -n '/^--- generation ---/,$p' "$OUT_TXT.raw" > "$OUT_TXT"
      rm -f "$OUT_TXT.raw"
      echo "═══ temp=$T top_p=$P prompt='$PROMPT' ═══" >> "$EVAL_LOG"
      cat "$OUT_TXT" >> "$EVAL_LOG"
      echo >> "$EVAL_LOG"
      if grep -qiE "$MARKER_PATTERNS" "$OUT_TXT"; then
          MARKER_HITS=$((MARKER_HITS + 1))
      fi
      if grep -qE '[一-龥]|[가-힣]|�' "$OUT_TXT"; then
          GARBAGE_HITS=$((GARBAGE_HITS + 1))
      fi
    done
  done
done
log "  multi-temp eval: $MARKER_HITS / $TOTAL marker hits, $GARBAGE_HITS garbage cells"
log "  full transcripts: $EVAL_LOG"

# ─── 7. Acceptance gate ──────────────────────────────────────────────────
log "─── step 7: acceptance gate ───"
PASS=1
if [ -z "$BEST_VAL" ]; then
    log "  best_val parse failed → FAIL"; PASS=0
else
    if awk "BEGIN{exit !($BEST_VAL < 1.5)}"; then
        log "  ✓ best_val=$BEST_VAL < 1.5"
    else
        log "  ✗ best_val=$BEST_VAL ≥ 1.5 → FAIL"; PASS=0
    fi
fi
if [ "$MARKER_HITS" -ge 5 ]; then
    log "  ✓ marker hits $MARKER_HITS ≥ 5"
else
    log "  ✗ marker hits $MARKER_HITS < 5 → FAIL"; PASS=0
fi
if [ "$GARBAGE_HITS" -le 2 ]; then
    log "  ✓ garbage cells $GARBAGE_HITS ≤ 2"
else
    log "  ✗ garbage cells $GARBAGE_HITS > 2 → FAIL"; PASS=0
fi

if [ "$PASS" -eq 0 ]; then
    log "════════════════════════════════════════════════════════"
    log "  TERMINAL FAIL — no upload, no theatrical PASS"
    log "════════════════════════════════════════════════════════"
    exit 16
fi
log "step 7 OK — acceptance gate PASSED"

# ─── 8. Upload to HF (use `hf` CLI, NOT deprecated huggingface-cli) ──────
log "─── step 8: upload to HF $HF_REPO ───"
if ! command -v hf >/dev/null 2>&1; then
    log "FATAL: hf CLI missing on pod"; exit 17
fi
hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true

hf_upload_check() {
    local src="$1" dst="$2"
    log "  uploading $dst..."
    hf upload "$HF_REPO" "$src" "$dst" --repo-type model 2>&1 | tail -3
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then log "FATAL: hf upload '$dst' exit $rc"; exit 18; fi
}
hf_upload_check "$LORA_USE"                                        "lora_v3/lora_arianna_v3_best.bin"
hf_upload_check "$OUT/slot_arianna/resonance_arianna_lora_v3_q8_0.gguf" "lora_v3/slot_arianna/resonance_arianna_lora_v3_q8_0.gguf"
hf_upload_check "$OUT/slot_arianna/resonance_arianna_lora_v3_q4_k.gguf" "lora_v3/slot_arianna/resonance_arianna_lora_v3_q4_k.gguf"
hf_upload_check "$RUN_REPORT"                                      "lora_v3/run_report_lora_v3.txt"
hf_upload_check "$EVAL_LOG"                                        "lora_v3/eval/multi_temp_eval.txt"
log "step 8 OK — HF artifacts pushed (all uploads verified)"

log "════════════════════════════════════════════════════════"
log "  ALL DONE — best_val=$BEST_VAL  markers=$MARKER_HITS/$TOTAL  garbage=$GARBAGE_HITS"
log "════════════════════════════════════════════════════════"
exit 0
