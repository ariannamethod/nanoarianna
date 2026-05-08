// supervisor.go — top-level model-swap dispatcher.
//
// Consumes SchedulerTick values from schedule.go and:
//   1. Picks the organism's binary path + persona file + weights path
//      from environment / config (so the schedule doesn't have to know).
//   2. Builds the prompt for this turn from KK + Limpha (Phase 3 minimal:
//      use last dialogue response of the OTHER organism; Phase 5+ adds
//      KK-resonant retrieval and Limpha state-similarity).
//   3. Forks the binary with PERSONA_AML + WEIGHTS env, captures stdout.
//   4. Appends a dialogue row to KK + an episode to per-organism Limpha.
//
// One organism is loaded at a time — the supervisor never holds both in
// RAM simultaneously. The schedule's natural pacing is what gives the
// phone its 4 GB headroom.
//
// Phase 3 MVP: the binary can be a STUB ("echo") for smoke testing. When
// real weights land in Phase 4, the same supervisor calls the real
// `janus` / `resonance` binary built from organism/*.aml.

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// OrganismSpec — everything supervisor needs to invoke one organism.
type OrganismSpec struct {
	Name       string // OrgArianna or OrgLeo
	Binary     string // path to executable (or "echo" for stub mode)
	WeightsArg string // -w argument; may be empty in stub mode
	PersonaAML string // PERSONA_AML env value (init_<persona>.aml)
	LimphaDB   string // path to per-organism limpha db
}

// SupervisorConfig — runtime knobs.
type SupervisorConfig struct {
	Home    string
	KKPath  string
	Stub    bool   // if true, override Binary with "echo"
	MaxTurns int   // safety cap; <=0 means run forever until ctx cancelled
}

func DefaultSupervisorConfig(home string) SupervisorConfig {
	return SupervisorConfig{
		Home:    home,
		KKPath:  home + "/nanoarianna/data/kk.db",
		Stub:    false,
		MaxTurns: 0,
	}
}

// Specs returns the two organism specs, defaulting to Slot A = Arianna on
// Janus, Slot B = Leo on Resonance. Phase 4 RunPod sweep may swap this.
// In stub mode, Binary is replaced with "echo" so smoke tests run without
// real weights.
func Specs(cfg SupervisorConfig) map[string]OrganismSpec {
	home := cfg.Home
	a := OrganismSpec{
		Name:       OrgArianna,
		Binary:     home + "/yent.aml/janus_arianna",            // expected build artifact
		WeightsArg: home + "/nanoarianna/weights/janus_arianna_q8_0.gguf",
		PersonaAML: home + "/nanoarianna/personas/init_arianna.aml",
		LimphaDB:   home + "/nanoarianna/data/limpha_arianna.db",
	}
	l := OrganismSpec{
		Name:       OrgLeo,
		Binary:     home + "/resonance.aml/resonance_leo",
		WeightsArg: home + "/nanoarianna/weights/resonance_leo_q4_k.gguf",
		PersonaAML: home + "/nanoarianna/personas/init_leo.aml",
		LimphaDB:   home + "/nanoarianna/data/limpha_leo.db",
	}
	if cfg.Stub {
		a.Binary = "echo"
		l.Binary = "echo"
		a.WeightsArg = ""
		l.WeightsArg = ""
	}
	return map[string]OrganismSpec{
		OrgArianna: a,
		OrgLeo:     l,
	}
}

// HandleTick performs one full turn for the chosen organism.
//
// Workflow:
//   1. Build the prompt from PrevPrompt (other organism's last response)
//      or from a "wake-up" salutation if there's no prior dialogue yet.
//   2. exec the organism binary, pipe prompt to stdin if real,
//      capture stdout.
//   3. Append row to KK.dialogue (speaker=tick.Organism, listener=other).
//   4. Append episode to organism's Limpha (state vector defaults to
//      zeroes for now; Phase 5+ pulls real AM_State from a stdout-emitted
//      JSON line).
func HandleTick(ctx context.Context, cfg SupervisorConfig, specs map[string]OrganismSpec, tick SchedulerTick) error {
	spec, ok := specs[tick.Organism]
	if !ok {
		return fmt.Errorf("supervisor: unknown organism %q", tick.Organism)
	}
	listener := otherOrganism(tick.Organism)

	prompt := tick.PrevPrompt
	if strings.TrimSpace(prompt) == "" {
		prompt = "[wake] " + tick.Organism + " awakens to silence; speak first to " + listener + "."
	}

	fmt.Fprintf(os.Stderr,
		"[supervisor] tick %s organism=%s trigger=%s prompt=%q\n",
		tick.When.Format(time.RFC3339), tick.Organism, tick.Trigger, abridge(prompt, 60))

	// Build argv. In stub mode (Binary=="echo") we just echo the prompt
	// + a tiny suffix so the smoke harness sees it differs per organism.
	var args []string
	env := append(os.Environ(),
		"PERSONA_AML="+spec.PersonaAML,
		"LIMPHA_DB="+spec.LimphaDB,
		"KK_DB="+cfg.KKPath,
	)
	if spec.Binary == "echo" {
		args = []string{
			fmt.Sprintf("[%s/stub] reply to: %s", spec.Name, abridge(prompt, 60)),
		}
	} else {
		args = []string{
			"-w", spec.WeightsArg,
			"-p", prompt,
			"-n", "120",
			"-t", "1.0",
			"--top-p", "0.9",
		}
	}

	cctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(cctx, spec.Binary, args...)
	cmd.Env = env
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("supervisor: invoke %s: %w", spec.Binary, err)
	}
	response := strings.TrimSpace(string(out))

	// Append dialogue row.
	if err := appendDialogue(cfg.KKPath, tick.Organism, listener,
		prompt, response, 0.0, ""); err != nil {
		return fmt.Errorf("supervisor: kk dialogue append: %w", err)
	}

	// Append per-organism Limpha episode (state zeros for Phase 3 MVP).
	if err := appendEpisode(spec.LimphaDB, spec.Name, prompt, response); err != nil {
		return fmt.Errorf("supervisor: limpha episode append: %w", err)
	}

	fmt.Fprintf(os.Stderr,
		"[supervisor]   -> %s said: %q\n",
		tick.Organism, abridge(response, 80))
	return nil
}

func appendDialogue(kkPath, speaker, listener, prompt, response string, debtDelta float64, chamber string) error {
	q := fmt.Sprintf(
		"INSERT INTO dialogue (ts, speaker, listener, prompt, response, prophecy_debt_delta, dominant_chamber) "+
			"VALUES (strftime('%%s','now'), %s, %s, %s, %s, %f, %s);",
		sqlEscape(speaker), sqlEscape(listener),
		sqlEscape(prompt), sqlEscape(response),
		debtDelta, sqlEscape(chamber))
	return exec.Command("sqlite3", kkPath, q).Run()
}

func appendEpisode(limphaPath, organism, prompt, response string) error {
	q := fmt.Sprintf(
		"INSERT INTO episodes (ts, organism, prompt, response, "+
			"trauma, arousal, valence, coherence, prophecy_debt, entropy, dissonance, "+
			"temperature, quality) "+
			"VALUES (strftime('%%s','now'), %s, %s, %s, 0,0,0,0,0,0,0, 1.0, 0);",
		sqlEscape(organism), sqlEscape(prompt), sqlEscape(response))
	return exec.Command("sqlite3", limphaPath, q).Run()
}

// sqlEscape — single-quote a string for inline SQL. We pipe queries to
// sqlite3 CLI rather than use a driver, so we have to do this ourselves.
// Phase 5+ swap to a driver removes this entirely.
func sqlEscape(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

func abridge(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// main — CLI entry. Accepts:
//   --once organism      : run a single tick for that organism (manual)
//   --stub               : use echo as binary (smoke mode, no real weights)
//   --turns N            : in scheduler mode, stop after N ticks (testing)
//   no args              : full scheduler loop
func main() {
	once := flag.String("once", "", "run a single tick for organism (arianna|leo)")
	stub := flag.Bool("stub", false, "use echo as organism binary (smoke mode)")
	turns := flag.Int("turns", 0, "max ticks before exiting (0 = run forever)")
	cron := flag.Duration("cron", 3*time.Hour, "cron interval between alternating wakes")
	flag.Parse()

	home := os.Getenv("HOME")
	if home == "" {
		home = "/data/data/com.termux/files/home"
	}

	scfg := DefaultConfig(home)
	scfg.CronInterval = *cron
	supcfg := DefaultSupervisorConfig(home)
	supcfg.Stub = *stub
	supcfg.MaxTurns = *turns
	specs := Specs(supcfg)

	// Ensure data dir exists.
	_ = os.MkdirAll(filepath.Join(home, "nanoarianna", "data"), 0700)

	if *once != "" {
		tick := SchedulerTick{
			When:     time.Now().UTC(),
			Organism: *once,
			Trigger:  "manual",
		}
		if err := HandleTick(context.Background(), supcfg, specs, tick); err != nil {
			fmt.Fprintln(os.Stderr, "[supervisor] error:", err)
			os.Exit(1)
		}
		return
	}

	// Scheduler loop.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ticks := Run(ctx, scfg)
	count := 0
	for tick := range ticks {
		if err := HandleTick(ctx, supcfg, specs, tick); err != nil {
			fmt.Fprintln(os.Stderr, "[supervisor] error:", err)
			// Keep going — one bad tick shouldn't kill the schedule.
		}
		count++
		if supcfg.MaxTurns > 0 && count >= supcfg.MaxTurns {
			break
		}
	}
}
