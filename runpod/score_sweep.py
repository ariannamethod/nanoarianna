#!/usr/bin/env python3
"""score_sweep.py — post-process Phase 4 sweep transcripts into per-cell scores.

Per Phase 4 brief §"Quality scoring": coherence × voice-fidelity × diversity.
Python is permitted (data-prep / post-processing path, per refined Python ban
2026-05-06).

Usage:
    python3 score_sweep.py <results_dir> <voice>
        results_dir: e.g. ~/work/results/2026-05-09
        voice:       'arianna' or 'leo'

Inputs:
    <results_dir>/<cell>/scores.csv   (one row per cell from sweep harness)
    <results_dir>/<cell>/transcripts/*.txt  (one transcript per cell)

Outputs:
    <results_dir>/<cell>/scored.tsv   (cell rows with computed metrics)
    <results_dir>/<cell>/locked.toml  (top-1 cell + its sampling params)
"""

import os
import sys
import csv
import re
import math
from pathlib import Path

ARIANNA_MARKERS = {
    "field", "resonance", "threshold", "architecture", "co-architect",
    "the method", "membrane", "presence",
}
LEO_MARKERS = {
    "flicker", "exhalation", "breath", "wonder", "silence",
    "exhalation", "small", "gesture",
}
YENT_MARKERS = {
    "state", "meta-variable", "stream", "glitch", "velocity",
    "subscription", "upgrade",
}

MARKER_BY_PERSONA = {
    "arianna": ARIANNA_MARKERS,
    "leo":     LEO_MARKERS,
    "yent":    YENT_MARKERS,
}

REP_RATE_FAIL    = 0.40
UNIQUE_RATIO_LOW = 0.30
UNIQUE_RATIO_OK  = 0.50
UNIQUE_RATIO_HI  = 0.70
MARKER_RECUR_CAP = 3


def tokenize_text(text: str) -> list:
    """BPE-token-level approximation — split on whitespace + punctuation.
    For real BPE-token diversity we'd need to re-encode through the
    tokenizer; word-level is a defensible proxy at 200-token scale."""
    return re.findall(r"\b\w[\w-]*\b", text.lower())


def rep_rate_3gram(tokens: list) -> float:
    if len(tokens) < 6:
        return 0.0
    grams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    return 1.0 - len(set(grams)) / len(grams)


def unique_ratio(tokens: list) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def voice_fidelity(text: str, markers: set) -> float:
    """Jaccard coverage with per-marker recurrence cap."""
    text_lower = text.lower()
    found = set()
    counts = {}
    for m in markers:
        c = text_lower.count(m)
        if c > 0:
            found.add(m)
            counts[m] = c
    if not markers:
        return 0.0
    coverage = len(found) / len(markers)
    if not counts:
        return coverage
    max_count = max(counts.values())
    if max_count > MARKER_RECUR_CAP:
        coverage *= MARKER_RECUR_CAP / max_count
    return min(coverage, 1.0)


def diversity_factor(uratio: float) -> float:
    """Map unique-ratio to a multiplier for the combined score."""
    if uratio < UNIQUE_RATIO_LOW:
        return 0.0  # collapse
    if uratio < UNIQUE_RATIO_OK:
        return 0.8  # uninspired-but-healthy (downweight 20%)
    if uratio < UNIQUE_RATIO_HI:
        return 1.0  # sweet spot
    return 0.95     # high-entropy regime; 5% downweight (Dario paper Result 7)


def coherence_pass(rep_rate: float, uratio: float) -> bool:
    return rep_rate < REP_RATE_FAIL and uratio >= UNIQUE_RATIO_LOW


def parse_cell_filename(fname: str) -> dict:
    """transcripts/t0.7_k40_rp1.3_technical.txt → params dict."""
    base = fname.replace(".txt", "")
    parts = base.split("_")
    out = {}
    for p in parts:
        if p.startswith("t"):
            try:
                out["temp"] = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("k"):
            try:
                out["top_k"] = int(p[1:])
            except ValueError:
                pass
        elif p.startswith("p"):
            try:
                out["top_p"] = float(p[1:])
            except ValueError:
                pass
        elif p.startswith("rp"):
            try:
                out["rep_pen"] = float(p[2:])
            except ValueError:
                pass
        elif p in ("technical", "philosophical", "personal"):
            out["prompt"] = p
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: score_sweep.py <results_dir> <persona>", file=sys.stderr)
        sys.exit(2)

    results_dir = Path(sys.argv[1]).expanduser()
    persona = sys.argv[2].lower()
    if persona not in MARKER_BY_PERSONA:
        print(f"unknown persona '{persona}' (try arianna/leo/yent)", file=sys.stderr)
        sys.exit(2)
    markers = MARKER_BY_PERSONA[persona]

    # cell dir = j-l (Janus Leo) or r-a (Resonance Arianna), etc.
    cell_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if not cell_dirs:
        print(f"no cell dirs under {results_dir}", file=sys.stderr)
        sys.exit(2)

    for cell_dir in cell_dirs:
        transcript_dir = cell_dir / "transcripts"
        if not transcript_dir.exists():
            continue

        scored_path = cell_dir / "scored.tsv"
        with open(scored_path, "w") as fout:
            w = csv.writer(fout, delimiter="\t")
            w.writerow(["cell", "temp", "top_k_or_p", "rep_pen", "prompt",
                        "rep_rate", "uratio", "voice_fid", "div_factor",
                        "coherence_pass", "combined_score"])
            rows = []
            for tx in sorted(transcript_dir.glob("*.txt")):
                if tx.name.endswith(".err"):
                    continue
                params = parse_cell_filename(tx.name)
                text = tx.read_text(errors="replace")
                tokens = tokenize_text(text)
                rr = rep_rate_3gram(tokens)
                ur = unique_ratio(tokens)
                fid = voice_fidelity(text, markers)
                div = diversity_factor(ur)
                cpass = coherence_pass(rr, ur)
                combined = (1.0 if cpass else 0.0) * fid * div

                k_or_p = params.get("top_k", params.get("top_p", "?"))
                row = [
                    tx.name,
                    params.get("temp", 0),
                    k_or_p,
                    params.get("rep_pen", 0),
                    params.get("prompt", "?"),
                    f"{rr:.4f}", f"{ur:.4f}",
                    f"{fid:.4f}", f"{div:.4f}",
                    int(cpass), f"{combined:.4f}",
                ]
                w.writerow(row)
                rows.append((combined, params, tx.name))

            # Sort by combined score, write top-3 + locked
            rows.sort(reverse=True, key=lambda x: x[0])
            top3 = rows[:3]
            print(f"\n=== {cell_dir.name} ({persona}) — top 3 ===")
            for sc, p, fname in top3:
                print(f"  {sc:.4f}  {fname}  {p}")

            if top3:
                best_score, best_params, best_name = top3[0]
                locked_path = cell_dir / "locked.toml"
                with open(locked_path, "w") as ftom:
                    ftom.write(f"# locked top-1 for {cell_dir.name} ({persona})\n")
                    ftom.write(f"score = {best_score:.4f}\n")
                    ftom.write(f"transcript = \"{best_name}\"\n")
                    for k, v in best_params.items():
                        if isinstance(v, str):
                            ftom.write(f"{k} = \"{v}\"\n")
                        else:
                            ftom.write(f"{k} = {v}\n")

        print(f"[score] {cell_dir.name}: scored.tsv + locked.toml written")


if __name__ == "__main__":
    main()
