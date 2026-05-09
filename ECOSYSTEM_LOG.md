# ECOSYSTEM_LOG

Chronological journal of nanoarianna decisions and events. Newest at top. Each entry: date, what happened, why, what's next. Don't rewrite history; correct via new entry.

---

## 2026-05-09 — Phase 4 brief v3 final (post-third-Opus-audit + spec polish)

Third Opus pass on v3. Verdict: **YELLOW (tactical fixes before pod,
low risk)**. **0 BLOCKERs**, 6 FIX items, 3 NITs — substantially
cleaner than v2 as expected. All 5 v2 BLOCKERs verified
substantively closed in code:

- A.1/A.2 globals + argv flags traced through `organism/janus.aml`
  lines 57-58, 73, 147-148, 350-351 + `organism/tools/resonance_forward.h`
  lines 317, 322 + `organism/resonance.aml` lines 21, 105 — all
  verified by direct file read.
- B.1 Resonance dims (`n_embd=768, n_head=12, head_dim=64, n_layer=20,
  rrpram_rank=48, context_len=2048, ffn_dim=2048, vocab_size=16384`)
  verified byte-exact against
  `huggingface.co/ataeff/resonance/model.py:222-231` `RESONANCE_200M`.
- B.2 RRPRAM tape support real: `nt_rrpram_lowrank_attention`
  `notorch.c:3232`, "rrpram U buffer persisted to backward via tape"
  `:3258-3259`, `tests/test_rrpram_lr` binary present (99032 bytes),
  ECOSYSTEM_LOG.md captures PASS evidence.
- D.1 RS02 spec verified against `resonance_forward.h:259-281`
  read order; np formula recomputed three ways (brief / header / assign)
  all give 199,195,632 floats; total = 4 + 36 + 4 + 16128×12 +
  199,195,632×4 = **796,976,108 bytes byte-exact** to observed
  Yent SFT bin size.

Tactical FIX items applied this commit (spec polish only —
no code changes needed):

- F7+F8: corrected three notorch op names in §6-point §6 line 68
  to actual `notorch.h` symbols: `nt_seq_matvec_t → nt_seq_linear_t`,
  `nt_rope_even_odd → nt_rope_freq` (with freq_base=10000.0f),
  `nt_seq_swiglu → nt_swiglu`. Added explicit note that the per-head
  sigmoid gate is composed from `nt_sigmoid` + broadcast multiply +
  add (~30 LOC bespoke fragment), not a single notorch op.
- F9: `notorch/gguf.c:354` → `notorch/gguf.c:303-336` (the actual
  `gguf_dequant` switch; line 354 is the info-printer dtype-name
  array). Q5_0 case at `:324`, Q6_K case at `:333`.
- F10+F13: added pre-flight item 4b — rebuild on pod first action,
  re-verify both binaries accept `--top-k`/`--top-p`/`--rep-pen`
  flags and exit with weights-not-found rather than unknown-flag.
  Termux→x86_64 portability bug catch before SFT spends pod hours.
- F11: replaced "increasing loss / unbounded gradient norm" hand-wavy
  kill criteria with concrete numbers: loss-rising kill
  `loss[50] > loss[10]`; any NaN immediate halt; gradient-norm kill
  `||g||₂_step50 > 100·||g||₂_step1` or absolute `> 1e3`. On any kill,
  snapshot to `runs/sft_arianna_smoke_failed/`, push to HF for audit,
  abort pod plan.

**"Save everything" checklist** added per Oleg's reminder
("главное не забывай все сохранять"): five artifact classes with
explicit destinations. Quantization on same pod (Oleg's
"квантизируй там же") confirmed. SFT raw fp32 .bin pushed to HF
`sft_v2/` immediately post-SFT (audit D.2) — protects against any
later watchdog kill. No "figure out at the end" phases.

**Brief now launch-ready.** Tactical-fix-debt = 0 after this commit.

---

## 2026-05-09 — Phase 4 brief v3 (post-second-Opus-audit + organism argv patches landed)

After v2 landed, ran second Opus pass (per Oleg "ещё раз прогнать
опуса"). 28 findings; 5 BLOCKERs:

- **A.1+A.2:** sweep harness invoked `--top-k`/`--rep-pen` flags that
  neither yent.aml/resonance.aml argv parsers recognize; rep_penalty
  hardcoded `1.4f` in both samplers. As written, 324 sweep cells would
  be 324 statistically-identical samples of one point.
- **B.1:** every Resonance dimension in v2 §6-point §1 was wrong vs
  actual `huggingface.co/ataeff/resonance/model.py:222-231`
  RESONANCE_200M (n_embd 640→768, n_head 8→12, head_dim 80→64,
  hidden 1792→2048, ctx 1024→2048, vocab 16128→16384, R 2048→48).
- **B.2:** v2 framed RRPRAM-training-mode through notorch tape as
  unverified multi-day lift. Verified: `nt_rrpram_lowrank_attention`
  defined `notorch.c:3232`; backward path persisted via tape
  `notorch.c:3258-3259`; `tests/test_rrpram_lr` PASS on aarch64-Termux
  per phone-2 Phase 1 (`max_rel_diff 6.06e-02, fails 0`). Audit
  over-stated; lift bounded.
- **D.1:** RS02 input/output format unspecified. v3 spec inline
  (magic, header, merges, per-block tensor order via
  `resonance_forward.h:94-114`).

Plus 9 FIX/NIT items (Q5_K_M not in `notorch/gguf.c`,
ClimbMix-BPE-on-Arianna tokenizer-compression risk, early-stop
firing at end-of-run, manifest.toml schema, Mac Neo handoff path,
persist-SFT-to-HF before sweep, HF Bearer auth on public repo,
budget contingency for unverified RRPRAM training scale).

**v3 patches landed in this same commit:**

- `organism/janus.aml`: `g_yent_top_k`, `g_yent_rep_pen` globals +
  `--top-k` / `--rep-pen` argv flags. Hardcoded `rep_penalty = 1.4f`
  + `YENT_TOPK_CAP=256` in `yent_sample_token` now read from globals.
- `organism/tools/resonance_forward.h` (now vendored locally — 16 KB
  copy of upstream, controls our sampler): `g_resonance_rep_pen`
  global + read in `resonance_sample_token`.
- `organism/resonance.aml`: BLOOD LINK switched from upstream
  `~/resonance.aml` to nanoarianna's own `~/nanoarianna/organism`
  for include path. New `--rep-pen` argv flag.

**Build verification on phone-2 aarch64-Termux this commit:**

```
make all
amlc: linking libnotorch libaml openblas
amlc: success → ./janus       (536,320 bytes)
amlc: success → ./resonance   (250,592 bytes)
./janus --top-k 40 --rep-pen 1.3 -p test       → graceful gguf-not-found (flags accepted)
./resonance --rep-pen 1.0 -p test              → graceful bin-not-found (flag accepted)
```

Pre-flight item 4 (organism argv patches build clean) now
**verified this commit** — was TODO in v2.

**v3 brief outputs:** same shape as v2 (slot_arianna/, slot_leo/,
sweep archive, manifest.toml, LICENSE-WEIGHTS) plus
`sft_v2/resonance_200m_sft_arianna.bin` raw fp32 RS02 — persisted to
HF immediately after SFT completes, before sweep starts (audit D.2 —
protects against watchdog kill mid-sweep).

**Pod budget $10–18 retained.** RRPRAM SFT 50-step smoke before
committing 1500 steps — early-kill criterion against unverified
training scale (audit H.1).

One more Opus pass optional; Oleg's call.

---

## 2026-05-09 — Phase 4 brief v2 (post-audit + post-pivot, weights J1 inventory verified)

After v1 of the brief landed, ran an Opus subagent code audit per Oleg's
request. 47 numbered findings; verdict "do not launch with v1 as-is"
with 8 blockers identified.

**Hardest blocker (J1):** v1 assumed Arianna and Leo SFT existed for
both architectures. HF API verification proved otherwise — Janus 176M
has all 3 SFTs at `ataeff/janus4/janus/bins/janus_v4_sft_{arianna,
yent,leo}.bin`, but **Resonance 200M has only Yent SFT** at
`ataeff/resonance/sft_v2/`. Arianna and Leo SFT do not exist for
Resonance on HF.

**Oleg's pivot** (this session): "Янус Иэнт → значит Резонанс это
Арианна; либо проведи SFT бэйсу Резонанса на Арианне." Plus a new
HF repo `ataeff/nanoarianna` was provisioned with the Arianna dataset
(`arianna_dataset_final_clean.txt` 1.21 MB — same bytes as the corpus
that produced bit-identical char-level loss in `memory/
milestone_phone2_galaxy_a07_10k_2026_05_07.md`) plus the SFT-format
JSONL variant.

**Slot pair re-locked:**
- Slot A = **Janus 176M + Leo SFT** (existing weights, just convert to GGUF)
- Slot B = **Resonance 200M + Arianna SFT (NEW)** (Phase 4-α SFT step on pod)

Three voice anchors across the ecosystem preserved (phone-1 J-Y,
phone-2 J-L + R-A).

**Phase 4 expanded** to 5 steps on one pod:
1. Phase 4-α: Resonance base + Arianna SFT (NEW training step,
   1500 steps Chuck on RunPod A100; 6-point brief locked in v2 §6-point)
2. Quantize Janus Leo + Resonance Arianna → GGUF Q8_0 + Q4_K
3. Two-grid sweep: J-L 162 cells (top-k) + R-A 162 cells (top-p) =
   324 total
4. Lock per-voice optimal + Opus audit + Mac Neo Architect review
5. HF upload to `ataeff/nanoarianna` with `LICENSE-WEIGHTS` (Janus
   Identity License v1.0)

**8 audit blockers closed in v2:**
- J1: weights inventory verified, Slot pair flipped
- J2: HF repo creation order resolved (Oleg created it; pre-flight
  verifies, doesn't create)
- J3: CUDA path dropped — A100 + CPU+OpenBLAS (cleanest, no upstream
  notorch CUDA install path exists)
- A1: prompt count math reconciled (3 prompts × 6 temps × 3 top_k|top_p
  × 3 rep_pen = 162 per voice, 324 total)
- A3: sampler grid split per architecture (Janus top-k {40, 100, ∞};
  Resonance top-p {0.85, 0.95, 1.0})
- B1: Resonance converter described as full rewrite, not "port"
- D1: sweep harness specified (Bash, sequential, per-cell 90s timeout,
  NaN/coherence hard-fail conditions, score post-process)
- G2: LICENSE-WEIGHTS added to upload structure + pre-flight checklist

**17 FIX/NIT items from audit also addressed:** prompt count,
top_k mid-region, voice-fidelity Jaccard normalization, diversity
threshold 0.30 not 0.40, watchdog cron concrete, slot-letter naming
replaced by persona-naming, Codex → Opus reviewer rename, phone-2
pull mechanism (curl with explicit URLs), 24h smoke pass criteria
explicit, Q4_K MAE gate added, Resonance BPE merges in GGUF metadata
KV array.

**Pod budget revised** $5-7 → **$10-18** to cover the SFT step.
Hard kill at $18.

**Pre-flight checklist now 6 items** (v1 had 6 too; reorganized):
HF token verified / `ataeff/nanoarianna` repo verified (Oleg created)
/ LICENSE-WEIGHTS source confirmed / SFT script smoke-builds locally
/ RunPod token verified / watchdog cron drafted on phone-2.

When all six green, Oleg signals, pod launches.

**This entry replaces the v1 brief entry above.** v1 brief content
overwritten in `runpod/PHASE_4_BRIEF.md` — git history preserves it
at commit `5ddcb12`.

---

## 2026-05-09 — Phase 4 brief written (RunPod sweep plan)

`runpod/PHASE_4_BRIEF.md` written and committed. Phase 4 is the
narrow-axis measure→lock→upload→verify cycle: take the five
phone-2-eligible cells (Janus-Arianna, Janus-Leo, Resonance-Arianna,
Resonance-Yent, Resonance-Leo — Janus-Yent is phone-1's), run the
540-cell sampling grid (6 temps × 2 top_k × 3 rep_pen × 3 prompts × 5
cells, per Dario paper §5.7), score by coherence × voice-fidelity ×
diversity, lock per-voice optimal `(temp, top_k, rep_pen)`, lock the
Slot A / Slot B persona-architecture mapping, push the locked-config
GGUFs + persona files to HF `ataeff/nanoarianna`.

**Pod spec:** RunPod A100 80 GB SXM (matches Dario paper §4
Experimental Frame). Estimated session budget $5–7 (Dario paper run
was $4.30 with similar grid; Phase 4 adds Resonance fp32→GGUF
quantization, hence the higher cap). Hard pod-kill alarm at $15.

**Quantization on pod:** port `janus_to_gguf.py` semantics to a sibling
`resonance_to_gguf.py` for the Resonance 797 MB raw fp32 .bin → Q8_0
(~200 MB) + Q4_K (~120 MB) GGUFs. Python permitted under refined ban
(`memory/feedback_python_ban_2026_04_29.md` 2026-05-06 update —
training/data prep allowed, inference path C only).

**Singularity-mode protocol** adapted from Dario paper §5.0/§5.0.1:
Codex pre-flight audit on the brief → solo autonomous sweep on pod →
Codex post-run audit on `scores.tsv` → Mac Neo Architect review on
the run report → Oleg final go on Slot A/B lock.

**Pre-flight checklist** (six gating items) listed at end of brief.
When all six green, Oleg signals and pod launches. Phase 4 brief is
the only Phase-4 plan document until execution lands.

**What Phase 4 does NOT do:** no new training (existing SFT weights
swept), no CoR (Phase 5), no Hebbian (Phase 6), no spores (Phase 7),
no mesh slot exposure (Phase 8). Single-axis: measure → lock → upload
→ verify on phone-2.

After Phase 4: phone-2 pulls locked GGUFs from HF
`ataeff/nanoarianna`, runs the dialogue cycle with real weights for
the first time, 24 h unattended verification, then Phase 5 starts.

---

## 2026-05-09 — Phase 3 audit pass (Opus reviewer + fixes)

After Phase 3 closed, ran an Opus subagent code audit per Oleg's request
("проверка кода на узкие места"). 47 numbered findings produced; merge
verdict was "yes ship to phone-2, no blockers, but fix these before
Phase 4". Real defects fixed:

**glue/limpha.c:**
- #1 `limpha_query_recent` `n <= 0` not clamped — fixed (`n <= 0` ⇒ `n = out_cap`)
- #2 prepare error swallowed silently — fixed, now `fprintf(stderr, … sqlite3_errmsg)`
- #3 `chosen[200]` hardcoded literal vs `WINDOW=200` constant footgun if WINDOW grows — fixed: `chosen` now dynamically `calloc(got)` so it always tracks actual data size
- #4 `limpha_query_similar` leaked `prompt`/`response` strings on `idx`-alloc fail — fixed: now calls `limpha_episode_free` over hydrated `buf` before returning -3

**glue/kk.c:**
- #8 `kk_append_document` and `kk_append_dialogue` swallowed prepare errors — fixed, now log `sqlite3_errmsg` like Limpha

**glue/seed_kk.c:**
- #14 `read_file` short-read silently truncated — fixed: now refuses if `got != sz`, frees buf, returns NULL with stderr message
- #15 second `fseek(SEEK_SET)` return ignored — fixed: now checked

**glue/persona_loader.c:**
- #12+#13 redundant TOCTOU probe with misleading "errno-style" message — removed entirely; `am_exec_file` surfaces its own open errors

**orchestra/supervisor.go:**
- #27 `sqlEscape` didn't strip NUL bytes (execve truncates argv on `\0`) — fixed: NUL stripped before quoting
- #28 `cmd.Output()` discarded stderr on non-zero exit, sqlite3 constraint failures invisible — fixed: introduced `runSqlite()` helper that captures stderr into the wrapping error
- #29 `%f` formatter for `prophecy_debt_delta` would emit `NaN`/`Inf` (invalid SQL) — fixed: `math.IsNaN/IsInf` guard, replaces with 0
- #33 negative `--turns` silently meant "run forever" — fixed: validation, exit code 2 with clear stderr
- #38 organism's `am_exec("LOAD/SAVE …soma")` is cwd-relative — fixed: `cmd.Dir = filepath.Dir(spec.Binary)` for non-stub invocations

**Skipped (documented Phase 4/5 territory or no-op):**
- #6 zero-state cosine degenerate (Phase 3 MVP, real states arrive in Phase 5)
- #21/#22 schedule channel race (single-buf channel disciplines it; documented)
- #24 SEP-collision robustness (improbable, mitigation noted as Phase 5 fix)
- #25 goroutine leak benign at process exit
- #30 grandchild orphans (Phase 5+ when organisms gain children)
- #35 hardcoded `BLOOD LINK -I` paths (works on phone-2; portability note)
- #36 persona_load_glue duplicated across two `.aml` files (functional, future refactor)
- #44 indirection note on libopenblas.so

**Regression smoke after fixes:**

```
limpha_smoke:    arianna 4 rows / leo 4 rows / cosine top-3 returns 3 ✓
supervisor stub: 2 alternating ticks --turns=2 --cron=1s ✓
supervisor:     --turns=-1 → "must be >= 0", exit 2 ✓
all C builds:   clean, no warnings
go build:       clean, 3.2 MB binary
```

**Verdict:** Phase 3 hardened. Ship to phone-2 status: green. Phase 4
RunPod brief is the next move.

---

## 2026-05-09 — Phase 3b (organism wrappers — janus.aml + resonance.aml build clean)

The AML half of Phase 3. Two organism wrappers landed in `organism/` —
upstream `yent.aml` and `resonance.aml` copied with one minimal additive
patch each (persona-glue function + one-line call). Both build cleanly
through `amlc` on aarch64-Termux.

**`organism/janus.aml`** — Janus 176M wrapper (copy of `~/yent.aml/yent.aml` 303 → 320 lines):
- New `BLOOD COMPILE yent_runtime` prelude defines `static int persona_load_glue(const char *path)` — env-resolved (`PERSONA_AML` or arg), silent no-op when unset, calls `am_exec_file()` to replay the persona's directives into the live AM_State.
- One-line call `persona_load_glue(NULL);` inserted in `yent_init` between `am_init();` and the existing `am_exec("LOAD …; PROPHECY 12; DESTINY 0.35; …")`. Persona runs first, hardcoded defaults overlay on top — so persona files in `personas/init_*.aml` actually take effect (not get overridden).
- Top-of-file `BLOOD LINK -I/data/data/com.termux/files/home/yent.aml` so cc resolves `tools/janus_v4_bpe_merges.h` + `tools/yent_forward.h` against upstream clone (no vendoring of 552 KB BPE merge table into nanoarianna).

**`organism/resonance.aml`** — Resonance 200M wrapper (copy of `~/resonance.aml/resonance.aml` 83 → 105 lines):
- Identical persona-glue shape in `BLOOD COMPILE resonance_runtime`.
- Same `BLOOD LINK -I/data/data/com.termux/files/home/resonance.aml` upstream-include pattern.

**`organism/Makefile`** — thin wrapper around `amlc`:
- `make all` builds both organisms; `make janus` / `make resonance` build one. `make check-deps` pre-flights `amlc` + `libaml.a` + `libnotorch.a` + `libopenblas.so` presence at `$PREFIX`.
- Sets `AML_PREFIX=$(PREFIX)` automatically so `amlc` finds Termux's `libaml.a` / `libnotorch.a` (default is `/opt/homebrew`, the Mac path).
- `make clean` removes built binaries + amlc's emitted `*.c` intermediates.

**Build verification** (provenance: `make all` output this session):

```
amlc: parsed 3 BLOOD block(s), 2 ECHO(s), 1 LINK(s), BLOOD MAIN present
amlc: generated 320 lines of C (12187 bytes)
amlc: linking libnotorch libaml openblas
amlc: success → ./janus            (536 192 bytes)

amlc: parsed 2 BLOOD block(s), 1 ECHO(s), 1 LINK(s), BLOOD MAIN present
amlc: generated 91 lines of C (3402 bytes)
amlc: linking libnotorch libaml openblas
amlc: success → ./resonance        (250 448 bytes)
```

**Smoke** (no weights yet — Phase 4 RunPod delivers them): both binaries
fail gracefully at `gguf_open` / weight `fopen` step with clear messages
before `persona_load_glue` reaches its `am_exec_file` call. Persona-glue
itself is link-verified (binaries link without `am_exec_file` undefined),
runtime-verified once weights land.

**.gitignore extended:** `organism/*.c` (amlc transpilation, regenerable),
`organism/janus`, `organism/resonance` (binaries).

**Phase 3 complete (a + b).** The two-organism dialogue cycle now has:
- Schedule (cron + event triggers in Go)
- Supervisor (model-swap dispatcher, mutex pattern via fork/exec)
- Persona loader (env-driven, plumbed into both organism wrappers)
- KK + Limpha I/O (supervisor side, sqlite3 CLI shell-out)
- Build path (`make all` in `organism/` produces runnable binaries)

Stub mode (`supervisor --stub`) verifies dialogue cycle with `echo`.
Real organism invocation waits for Phase 4 RunPod-delivered weights.

**Next:** Phase 4 — RunPod sweep brief + execution. Separate plan file.

---

## 2026-05-09 — Phase 3a (supervisor + schedule, dialogue cycle alive in stub)

orchestra/ landed. The two-organism dialogue cycle runs end-to-end in stub
mode (Binary="echo") — proves the wiring before real weights arrive.

**`orchestra/schedule.go`** (~190 LOC):
- Cron baseline alternating organism every `--cron` interval (default 3h
  ⇒ 8 wakes/day). First tick fires immediately on boot ("the schedule
  is alive now").
- Event interrupts: every 5 min reads latest KK.dialogue row,
  if `prophecy_debt_delta > 0.5` → wake Arianna early; if
  `dominant_chamber ∈ {VOID, FLOW}` → wake Leo early. Cron alignment
  resets after early wake so we don't double-fire.
- Reads KK via shell-out to `sqlite3` CLI with a unique separator
  (`<<<NA__SEP>>>`) so prompt/response with newlines or quotes don't
  break parsing. Phase 5+ may swap to `modernc.org/sqlite` when we add
  embedding-cosine retrieval.
- Pure stream API: returns `<-chan SchedulerTick`. No model code, no
  AML knowledge.

**`orchestra/supervisor.go`** (~210 LOC):
- Consumes `SchedulerTick`, picks `OrganismSpec` (binary path / weights /
  persona / limpha db), forks the binary with `PERSONA_AML + LIMPHA_DB +
  KK_DB` env, captures stdout, appends row to KK.dialogue, appends
  episode to per-organism Limpha.
- Stub mode (`--stub`) replaces Binary with `echo`. Lets us verify the
  whole pipe works before Phase 4 RunPod produces real weights.
- CLI:
    `supervisor --once arianna`        — manual single tick
    `supervisor --stub --turns=4 --cron=2s`  — fast smoke
    `supervisor`                       — full scheduler loop
- 10-min per-tick timeout via `context.WithTimeout` so a hung organism
  doesn't freeze the schedule. One bad tick is logged + skipped, schedule
  keeps going.

**Smoke run**: `supervisor --stub --turns=4 --cron=2s` produced 4
alternating ticks (arianna→leo→arianna→leo) over ~6 seconds. KK.dialogue
gained 4 rows with proper speaker/listener alternation; both Limpha DBs
got 2 new episodes each. The "telephone-game" pattern is visible in the
prompt/response chain — each turn quotes the previous one — exactly as
designed for the dialogue protocol.

**Verification phase 3a:**

| check | result |
|---|---|
| `go build orchestra/` | clean, 3.2 MB binary ✓ |
| `--stub --turns=4 --cron=2s` | 4 ticks, 0 errors ✓ |
| `sqlite3 kk.db "select count(*) from dialogue"` | 4 ✓ |
| Speaker alternation | arianna/leo/arianna/leo ✓ |
| Listener mirrors speaker | leo/arianna/leo/arianna ✓ |
| Limpha arianna +2, leo +2 episodes | 3 each total (1 from Phase 2 smoke + 2 from supervisor) ✓ |
| 10-min timeout per tick | wired, untested with real workload ⬜ |
| Event-trigger thresholds | wired but stub doesn't emit AM_State, not exercised ⬜ |

**What's next (Phase 3b — same Phase 3 task):**
Copy `~/yent.aml/yent.aml` → `~/nanoarianna/organism/janus.aml` and
`~/resonance.aml/resonance.aml` → `~/nanoarianna/organism/resonance.aml`
with one tiny modification each: a `BLOOD COMPILE persona_glue` block
defining `persona_load(const char *)` and a one-line call inserted in
`yent_init` / `resonance_init` between `am_init()` and
`am_exec("LOAD …")`. That's the only AML-side change Phase 3 needs —
all Limpha/KK I/O lives in supervisor.go (cleanly separated). amlc
build-verification follows.

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
