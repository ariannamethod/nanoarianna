#!/bin/bash
# run_sweep.sh — Phase 4 sweep harness. Two grids, sequential, no parallelism.
# 162 cells per voice × 2 voices = 324 cells. Per-cell timeout 90s.
#
# Hard-fail conditions (exit codes per audit F.2):
#   42 — NaN storm: 5 contiguous failures
#   43 — coherence streak: 3 consecutive cells with rep_rate > 0.5 OR unique_ratio < 0.30
#
# Inputs (env or argv):
#   RESULTS_DIR  — base output dir (default ~/work/results/$(date +%F))
#   JANUS_BIN    — path to Janus binary (default ~/work/nanoarianna/organism/janus)
#   RES_BIN      — path to Resonance binary (default ~/work/nanoarianna/organism/resonance)
#   LEO_GGUF     — path to Janus Leo Q8_0 GGUF
#   ARI_GGUF     — path to Resonance Arianna Q8_0 GGUF
#   PROMPTS_DIR  — dir with technical.txt / philosophical.txt / personal.txt
#
# Each cell writes a transcript file + appends a row to scores.csv.

set -u

RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/$(date +%F)}"
JANUS_BIN="${JANUS_BIN:-$HOME/work/nanoarianna/organism/janus}"
RES_BIN="${RES_BIN:-$HOME/work/nanoarianna/organism/resonance}"
LEO_GGUF="${LEO_GGUF:-$HOME/work/dist/slot_leo/janus_v4_leo_q8_0.gguf}"
ARI_GGUF="${ARI_GGUF:-$HOME/work/dist/slot_arianna/resonance_v2_arianna_q8_0.gguf}"
PROMPTS_DIR="${PROMPTS_DIR:-$HOME/work/nanoarianna/runpod/prompts}"
PERSONA_LEO="${PERSONA_LEO:-$HOME/work/nanoarianna/personas/init_leo.aml}"
PERSONA_ARI="${PERSONA_ARI:-$HOME/work/nanoarianna/personas/init_arianna.aml}"

CELL_TIMEOUT="${CELL_TIMEOUT:-90}"
N_TOKENS="${N_TOKENS:-200}"

mkdir -p "$RESULTS_DIR/j-l/transcripts" "$RESULTS_DIR/r-a/transcripts"
echo "temp,top_k,rep_pen,prompt,exit_code,wall_sec" > "$RESULTS_DIR/j-l/scores.csv"
echo "temp,top_p,rep_pen,prompt,exit_code,wall_sec" > "$RESULTS_DIR/r-a/scores.csv"

NAN_STREAK=0
COH_STREAK=0
NAN_LIMIT=5
COH_LIMIT=3

run_cell() {
    local label="$1"; shift
    local cmd_args="$1"; shift
    local out_file="$1"; shift

    local t0=$(date +%s)
    timeout "$CELL_TIMEOUT" bash -c "$cmd_args" > "$out_file" 2>"${out_file}.err"
    local rc=$?
    local t1=$(date +%s)
    local wall=$((t1 - t0))

    # NaN detection: organism prints "NaN" on stderr if forward-pass produces non-finite
    if [ $rc -ne 0 ] || grep -qiE "nan|inf" "${out_file}.err" 2>/dev/null; then
        NAN_STREAK=$((NAN_STREAK + 1))
        echo "[sweep $label] NaN/fail (streak=$NAN_STREAK)" >&2
    else
        NAN_STREAK=0
    fi

    if [ $NAN_STREAK -ge $NAN_LIMIT ]; then
        echo "[sweep] HARD FAIL — NaN storm $NAN_STREAK contiguous" >&2
        exit 42
    fi

    # Quick coherence check (heavy work in score_sweep.py post-pass)
    if [ -s "$out_file" ]; then
        local nbytes=$(wc -c < "$out_file")
        local nuniq=$(tr ' ' '\n' < "$out_file" | sort -u | wc -l)
        # ratio = unique words / total words; collapse if very low
        local ntot=$(tr ' ' '\n' < "$out_file" | wc -l)
        if [ "$ntot" -gt 0 ]; then
            local ratio=$(awk -v u="$nuniq" -v t="$ntot" 'BEGIN{printf "%.3f", u/t}')
            if awk -v r="$ratio" 'BEGIN{exit !(r < 0.30)}'; then
                COH_STREAK=$((COH_STREAK + 1))
            else
                COH_STREAK=0
            fi
        fi
    fi
    if [ $COH_STREAK -ge $COH_LIMIT ]; then
        echo "[sweep] HARD FAIL — coherence streak $COH_STREAK contiguous" >&2
        exit 43
    fi

    echo "$rc,$wall"
}

# ─── J-L grid: top-k + top-p + rep-pen ───────────────────────────────────
echo "[sweep] J-L grid begin: $(date +%FT%T)"
for prompt_id in technical philosophical personal; do
  prompt_file="$PROMPTS_DIR/${prompt_id}.txt"
  for t in 0.5 0.7 0.8 0.9 1.0 1.1; do
    for k in 40 100 256; do
      for rp in 1.0 1.3 1.4; do
        cell="$RESULTS_DIR/j-l/transcripts/t${t}_k${k}_rp${rp}_${prompt_id}.txt"
        cmd="PERSONA_AML='$PERSONA_LEO' '$JANUS_BIN' \
             -w '$LEO_GGUF' -p \"\$(cat '$prompt_file')\" \
             -n $N_TOKENS -t $t --top-p 0.9 --top-k $k --rep-pen $rp"
        out=$(run_cell "j-l/$t/$k/$rp/$prompt_id" "$cmd" "$cell")
        rc=$(echo "$out" | cut -d, -f1)
        wall=$(echo "$out" | cut -d, -f2)
        echo "$t,$k,$rp,$prompt_id,$rc,$wall" >> "$RESULTS_DIR/j-l/scores.csv"
      done
    done
  done
done
echo "[sweep] J-L grid done: $(date +%FT%T)"

# ─── R-A grid: top-p + rep-pen ─────────────────────────────────────────────
echo "[sweep] R-A grid begin: $(date +%FT%T)"
for prompt_id in technical philosophical personal; do
  prompt_file="$PROMPTS_DIR/${prompt_id}.txt"
  for t in 0.5 0.7 0.8 0.9 1.0 1.1; do
    for p in 0.85 0.95 1.0; do
      for rp in 1.0 1.3 1.4; do
        cell="$RESULTS_DIR/r-a/transcripts/t${t}_p${p}_rp${rp}_${prompt_id}.txt"
        cmd="PERSONA_AML='$PERSONA_ARI' '$RES_BIN' \
             -w '$ARI_GGUF' -p \"\$(cat '$prompt_file')\" \
             -n $N_TOKENS -t $t --top-p $p --rep-pen $rp"
        out=$(run_cell "r-a/$t/$p/$rp/$prompt_id" "$cmd" "$cell")
        rc=$(echo "$out" | cut -d, -f1)
        wall=$(echo "$out" | cut -d, -f2)
        echo "$t,$p,$rp,$prompt_id,$rc,$wall" >> "$RESULTS_DIR/r-a/scores.csv"
      done
    done
  done
done
echo "[sweep] R-A grid done: $(date +%FT%T)"

echo "[sweep] all 324 cells complete: $(date +%FT%T)"
