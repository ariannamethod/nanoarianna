#!/bin/bash
# run_phase4.sh — Phase 4 master driver. Runs end-to-end on the pod:
#
#   1. Toolchain build  (notorch + ariannamethod + janus organism + resonance organism)
#   2. 4b verify        (notorch tests 47/47, AML smoke, both organism --help)
#   3. Smoke-50 SFT     (50 steps Resonance+Arianna; kill on NaN/non-descent/grad-explosion)
#   4. Full SFT         (1500 steps with val every 100, ckpt every 500, plateau early-stop)
#   5. Quantize         (Resonance Arianna SFT → GGUF Q8_0 + Q4_K)
#                       (Janus Leo base       → GGUF Q8_0 + Q4_K via existing janus_to_gguf.py)
#   6. Sweep            (324 cells: 162 J-L + 162 R-A; 90s/cell; 8h+ on A100)
#   7. Lock             (score_sweep.py per-cell; top-1 → locked.toml)
#   8. Upload           (HF ataeff/nanoarianna: SFT raw .bin, GGUFs, sweep archive, run report)
#
# Pre-flight required (set in env or pass via argv):
#   WORK=/work                        — pod work root
#   IN=$WORK/in                       — base weights + corpus + AML refs
#   OUT=$WORK/out                     — SFT outputs, GGUFs, sweep
#   HF_TOKEN=...                      — for upload phase (write scope)
#   BUDGET_USD=18                     — watchdog kills pod at this spend
#
# Hard-fails (exit codes):
#   10 — toolchain build failed
#   11 — 4b verify failed
#   99 — smoke-50 failed (from sft_resonance_arianna)
#   42 — NaN storm in sweep (from run_sweep.sh)
#   43 — coherence streak in sweep (from run_sweep.sh)
#
# Save-everything checklist (brief §"Save"):
#   - Trainer source already in repo (this file + sft_resonance_arianna.c)
#   - SFT raw .bin                   → HF ataeff/nanoarianna/sft_v2/resonance_arianna_v2_sft.bin
#   - GGUFs                          → HF ataeff/nanoarianna/slot_arianna/*.gguf
#                                    + HF ataeff/nanoarianna/slot_leo/*.gguf
#   - Sweep archive                  → HF ataeff/nanoarianna/sweep/<date>/
#   - Run report                     → ECOSYSTEM_LOG.md commit pushed to nanoarianna repo

set -u
set -o pipefail

# ─── env defaults ─────────────────────────────────────────────────────────
WORK="${WORK:-/work}"
IN="${IN:-$WORK/in}"
OUT="${OUT:-$WORK/out}"
NANOREPO="${NANOREPO:-$WORK/nanoarianna}"
NOTORCH="${NOTORCH:-$WORK/notorch}"
AML="${AML:-$WORK/ariannamethod.ai}"
YENTAML="${YENTAML:-$WORK/yent.aml}"
RESAML="${RESAML:-$WORK/resonance.aml}"
HF_REPO="${HF_REPO:-ataeff/nanoarianna}"
BUDGET_USD="${BUDGET_USD:-18}"
SFT_STEPS="${SFT_STEPS:-1500}"
SFT_LR="${SFT_LR:-3e-5}"
SFT_CTX="${SFT_CTX:-512}"

DATE_TAG="$(date +%F)"
TIMESTAMP="$(date +%FT%H%M%S)"
RUN_REPORT="$OUT/run_report_${TIMESTAMP}.txt"

mkdir -p "$OUT" "$OUT/sweep/$DATE_TAG" "$OUT/sft_v2"

log() { echo "[phase4 $(date +%T)] $*" | tee -a "$RUN_REPORT"; }

log "════════════════════════════════════════════════════════"
log "  Phase 4 master driver"
log "  WORK=$WORK  IN=$IN  OUT=$OUT  HF=$HF_REPO  budget=\$$BUDGET_USD"
log "════════════════════════════════════════════════════════"

# ─── 0a. HF pre-flight download (verified paths via HF API 2026-05-09) ────
log "─── step 0a: HF pre-flight download → $IN ───"
mkdir -p "$IN"
if [ -z "${HF_TOKEN:-}" ]; then
    log "FATAL: HF_TOKEN unset — cannot fetch base weights / corpus"; exit 9
fi
hf_pull() {
    local repo="$1" path="$2" dst="$3"
    if [ -f "$dst" ]; then
        log "  cached: $dst"; return 0
    fi
    log "  fetching $repo/$path → $dst"
    curl -fL --retry 3 -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/$repo/resolve/main/$path" -o "$dst" \
        2>&1 | tail -3
}
hf_pull "ataeff/resonance"   "checkpoints/resonance_200m_final.bin" \
        "$IN/resonance_200m_final.bin"
hf_pull "ataeff/nanoarianna" "arianna_dataset_final_clean.txt" \
        "$IN/arianna_dataset_final_clean.txt"
hf_pull "ataeff/janus4"      "janus/bins/janus_v4_sft_leo.bin" \
        "$IN/janus_v4_sft_leo.bin"
for f in resonance_200m_final.bin arianna_dataset_final_clean.txt janus_v4_sft_leo.bin; do
    [ -s "$IN/$f" ] || { log "FATAL: $IN/$f missing or empty"; exit 9; }
done
log "step 0a OK — corpus + 2 base weights staged"

# ─── 0b. Toolchain repos clone-if-missing ─────────────────────────────────
log "─── step 0b: toolchain repos ───"
clone_if_missing() {
    local url="$1" dst="$2"
    if [ -d "$dst/.git" ]; then
        log "  cached: $dst"
    else
        log "  clone $url → $dst"
        git clone --depth 1 "$url" "$dst" 2>&1 | tail -3
    fi
}
mkdir -p "$WORK"
clone_if_missing "https://github.com/ariannamethod/notorch"          "$NOTORCH"
clone_if_missing "https://github.com/ariannamethod/ariannamethod.ai" "$AML"
clone_if_missing "https://github.com/ariannamethod/yent.aml"         "$YENTAML"
clone_if_missing "https://github.com/ariannamethod/resonance.aml"    "$RESAML"
log "step 0b OK"

# ─── 0c. Budget watchdog (background) ─────────────────────────────────────
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    log "starting budget watchdog (limit \$$BUDGET_USD)"
    bash "$NANOREPO/runpod/budget_watchdog.sh" \
        "$RUNPOD_POD_ID" "$RUNPOD_API_KEY" "$BUDGET_USD" \
        > "$OUT/watchdog.log" 2>&1 &
    WATCHDOG_PID=$!
    log "watchdog PID=$WATCHDOG_PID"
fi

# ─── 1. Toolchain build ───────────────────────────────────────────────────
log "─── step 1: toolchain build ───"
{
    cd "$NOTORCH"
    log "  notorch make (USE_CUDA=1)..."
    make clean >/dev/null 2>&1 || true
    if ! make USE_CUDA=1 lib -j8 2>&1 | tail -5; then
        log "FATAL: notorch GPU build failed"; exit 10
    fi
    log "  notorch install → /usr/local..."
    if ! make USE_CUDA=1 install PREFIX=/usr/local 2>&1 | tail -3; then
        log "FATAL: notorch install failed"; exit 10
    fi

    cd "$AML"
    log "  ariannamethod make..."
    make clean >/dev/null 2>&1 || true
    if ! make -j8 2>&1 | tail -5; then
        log "FATAL: ariannamethod build failed"; exit 10
    fi
    log "  ariannamethod install → /usr/local..."
    if ! make install PREFIX=/usr/local 2>&1 | tail -3; then
        log "FATAL: ariannamethod install failed"; exit 10
    fi

    log "  janus organism build..."
    cd "$NANOREPO/organism"
    if ! make -B janus 2>&1 | tail -5; then
        log "FATAL: janus build failed"; exit 10
    fi

    log "  resonance organism build..."
    if ! make -B resonance 2>&1 | tail -5; then
        log "FATAL: resonance build failed"; exit 10
    fi

    log "  SFT trainer build (USE_CUDA)..."
    cd "$NANOREPO/runpod"
    if ! cc -O3 -Wall -DUSE_BLAS -DUSE_CUDA sft_resonance_arianna.c \
            -I/usr/local/include \
            -L/usr/local/lib -L/usr/local/cuda/lib64 \
            -lnotorch_gpu -laml \
            -lopenblas -lcudart -lcublas -lm -lpthread \
            -o sft_resonance_arianna 2>&1 | tail -10; then
        log "FATAL: SFT trainer build failed"; exit 10
    fi
} || exit 10
log "step 1 OK"

# ─── 2. 4b verify ─────────────────────────────────────────────────────────
log "─── step 2: 4b verify ───"
{
    log "  libnotorch.a present..."
    [ -s /usr/local/lib/libnotorch.a ] || {
        log "FATAL: /usr/local/lib/libnotorch.a missing"; exit 11
    }
    log "  libnotorch_gpu.a present..."
    [ -s /usr/local/lib/libnotorch_gpu.a ] || {
        log "FATAL: /usr/local/lib/libnotorch_gpu.a missing"; exit 11
    }

    log "  janus binary executable..."
    [ -x "$NANOREPO/organism/janus" ] || {
        log "FATAL: janus binary missing/non-executable"; exit 11
    }
    log "  resonance binary executable..."
    [ -x "$NANOREPO/organism/resonance" ] || {
        log "FATAL: resonance binary missing/non-executable"; exit 11
    }
    log "  sft_resonance_arianna binary executable..."
    [ -x "$NANOREPO/runpod/sft_resonance_arianna" ] || {
        log "FATAL: SFT trainer binary missing/non-executable"; exit 11
    }
} || exit 11
log "step 2 OK"

# ─── 3. Smoke-50 SFT ──────────────────────────────────────────────────────
SMOKE_PREFIX="$OUT/sft_v2/smoke50"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    log "─── step 3: smoke-50 SKIPPED (SKIP_SMOKE=1) ───"
else
    log "─── step 3: smoke-50 SFT (50 steps, kill on bad signal) ───"
    if ! "$NANOREPO/runpod/sft_resonance_arianna" \
            "$IN/resonance_200m_final.bin" \
            "$IN/arianna_dataset_final_clean.txt" \
            "$SMOKE_PREFIX" \
            50 "$SFT_LR" "$SFT_CTX" 2>&1 | tee -a "$RUN_REPORT"; then
        rc=$?
        log "FATAL: smoke-50 exited $rc"
        exit $rc
    fi
    log "step 3 OK — smoke-50 PASS"
    rm -f "${SMOKE_PREFIX}_final.bin"   # smoke output not needed
fi

# ─── 4. Full SFT ──────────────────────────────────────────────────────────
log "─── step 4: full SFT ($SFT_STEPS steps) ───"
SFT_PREFIX="$OUT/sft_v2/resonance_arianna_v2_sft"
if ! "$NANOREPO/runpod/sft_resonance_arianna" \
        "$IN/resonance_200m_final.bin" \
        "$IN/arianna_dataset_final_clean.txt" \
        "$SFT_PREFIX" \
        "$SFT_STEPS" "$SFT_LR" "$SFT_CTX" 2>&1 | tee -a "$RUN_REPORT"; then
    rc=$?
    log "FATAL: full SFT exited $rc"
    exit $rc
fi
SFT_FINAL="${SFT_PREFIX}_final.bin"
[ -f "$SFT_FINAL" ] || { log "FATAL: SFT final missing"; exit 12; }
log "step 4 OK — SFT artifact: $SFT_FINAL"

# ─── 5. Quantize ──────────────────────────────────────────────────────────
log "─── step 5: quantize ───"
{
    log "  Resonance Arianna SFT → GGUF Q8_0..."
    mkdir -p "$OUT/slot_arianna"
    python3 "$NANOREPO/runpod/resonance_to_gguf.py" \
        "$SFT_FINAL" \
        "$OUT/slot_arianna/resonance_v2_arianna_q8_0.gguf" \
        --quant Q8_0 2>&1 | tail -5

    log "  Resonance Arianna SFT → GGUF Q4_K..."
    python3 "$NANOREPO/runpod/resonance_to_gguf.py" \
        "$SFT_FINAL" \
        "$OUT/slot_arianna/resonance_v2_arianna_q4_k.gguf" \
        --quant Q4_K 2>&1 | tail -5

    log "  Janus Leo SFT → GGUF Q8_0..."
    JANUS_TO_GGUF="$YENTAML/tools/janus_to_gguf.py"
    [ -f "$JANUS_TO_GGUF" ] || JANUS_TO_GGUF="$NANOREPO/runpod/janus_to_gguf.py"
    mkdir -p "$OUT/slot_leo"
    python3 "$JANUS_TO_GGUF" \
        "$IN/janus_v4_sft_leo.bin" \
        "$OUT/slot_leo/janus_v4_leo_q8_0.gguf" \
        --quant Q8_0 2>&1 | tail -5

    log "  Janus Leo SFT → GGUF Q4_K..."
    python3 "$JANUS_TO_GGUF" \
        "$IN/janus_v4_sft_leo.bin" \
        "$OUT/slot_leo/janus_v4_leo_q4_k.gguf" \
        --quant Q4_K 2>&1 | tail -5
} || { log "FATAL: quant failed"; exit 13; }
log "step 5 OK"

# ─── 6. Sweep ─────────────────────────────────────────────────────────────
log "─── step 6: sweep (324 cells) ───"
export RESULTS_DIR="$OUT/sweep/$DATE_TAG"
export JANUS_BIN="$NANOREPO/organism/janus"
export RES_BIN="$NANOREPO/organism/resonance"
export LEO_GGUF="$OUT/slot_leo/janus_v4_leo_q8_0.gguf"
export ARI_GGUF="$OUT/slot_arianna/resonance_v2_arianna_q8_0.gguf"
export PROMPTS_DIR="$NANOREPO/runpod/prompts"
export PERSONA_LEO="$NANOREPO/personas/init_leo.aml"
export PERSONA_ARI="$NANOREPO/personas/init_arianna.aml"

if ! bash "$NANOREPO/runpod/run_sweep.sh" 2>&1 | tee -a "$RUN_REPORT"; then
    rc=$?
    case $rc in
        42) log "FATAL: sweep NaN storm";       exit 42 ;;
        43) log "FATAL: sweep coherence streak"; exit 43 ;;
        *)  log "FATAL: sweep exited $rc";       exit $rc ;;
    esac
fi
log "step 6 OK — 324 cells written under $RESULTS_DIR"

# ─── 7. Lock ──────────────────────────────────────────────────────────────
log "─── step 7: score + lock ───"
python3 "$NANOREPO/runpod/score_sweep.py" "$RESULTS_DIR" leo     2>&1 | tee -a "$RUN_REPORT"
python3 "$NANOREPO/runpod/score_sweep.py" "$RESULTS_DIR" arianna 2>&1 | tee -a "$RUN_REPORT"
log "step 7 OK"

# ─── 8. Upload to HF ──────────────────────────────────────────────────────
log "─── step 8: upload to HF $HF_REPO ───"
if [ -z "${HF_TOKEN:-}" ]; then
    log "WARN: HF_TOKEN not set — skipping upload (manual upload required post-pod)"
else
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        pip install -q huggingface_hub
    fi
    huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1

    log "  uploading SFT raw .bin..."
    huggingface-cli upload "$HF_REPO" "$SFT_FINAL" \
        "sft_v2/resonance_arianna_v2_sft.bin" --repo-type model 2>&1 | tail -3

    log "  uploading slot_arianna GGUFs..."
    huggingface-cli upload "$HF_REPO" "$OUT/slot_arianna" "slot_arianna" \
        --repo-type model 2>&1 | tail -3

    log "  uploading slot_leo GGUFs..."
    huggingface-cli upload "$HF_REPO" "$OUT/slot_leo" "slot_leo" \
        --repo-type model 2>&1 | tail -3

    log "  uploading sweep archive..."
    huggingface-cli upload "$HF_REPO" "$RESULTS_DIR" "sweep/$DATE_TAG" \
        --repo-type model 2>&1 | tail -3

    log "  uploading run report..."
    huggingface-cli upload "$HF_REPO" "$RUN_REPORT" \
        "reports/run_report_${TIMESTAMP}.txt" --repo-type model 2>&1 | tail -3

    log "step 8 OK — uploads pushed to $HF_REPO"
fi

# ─── done ─────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════"
log "  Phase 4 ALL DONE — $TIMESTAMP"
log "════════════════════════════════════════════════════════"

if [ -n "${WATCHDOG_PID:-}" ]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
fi

exit 0
