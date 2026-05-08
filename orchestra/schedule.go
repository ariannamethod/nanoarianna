// schedule.go — when does which organism wake up.
//
// Two clocks running in parallel:
//
//   1. Cron baseline — alternating organism every cron_interval (default 3h,
//      so 8 wakes/day). This guarantees forward motion even when nothing
//      is going on field-physics-wise.
//
//   2. Event interrupt — read latest KK dialogue row, inspect
//      prophecy_debt_delta + dominant_chamber. If a threshold trips,
//      override the next cron tick and wake the matching organism early.
//
// Schedule sits between supervisor and the world. It produces a stream of
// SchedulerTick values; supervisor consumes one tick at a time and runs
// the actual model swap + invocation.
//
// Build (manual until we add a top-level Makefile):
//   cd ~/nanoarianna/orchestra && go build -o "$TMPDIR/schedule" \
//       schedule.go supervisor.go
//
// Per the plan, schedule.go has no model code or AML knowledge — it only
// knows organism *names* and reads the KK to make decisions. The actual
// model invocation lives in supervisor.go.

package main

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Organism names. Synced with KK.dialogue.speaker / .listener and with
// the binary names produced by the AML build (organism/janus.aml ->
// "janus", organism/resonance.aml -> "resonance"). Persona is set
// independently via env at invocation time.
const (
	OrgArianna = "arianna" // currently running on Janus 176M (Slot A default)
	OrgLeo     = "leo"     // currently running on Resonance 200M (Slot B default)
)

// SchedulerConfig — knobs read from schedule.toml or env. Keeping it
// dead simple in Phase 3a; refactor to TOML later.
type SchedulerConfig struct {
	CronInterval     time.Duration // default 3h => 8 wakes/day
	KKPath           string        // ~/nanoarianna/data/kk.db
	ProphecyDebtTrig float64       // event threshold for waking Arianna early
	VoidFlowTrig     bool          // if dominant_chamber ∈ {VOID, FLOW} → wake Leo early
}

func DefaultConfig(home string) SchedulerConfig {
	return SchedulerConfig{
		CronInterval:     3 * time.Hour,
		KKPath:           home + "/nanoarianna/data/kk.db",
		ProphecyDebtTrig: 0.5,
		VoidFlowTrig:     true,
	}
}

// SchedulerTick — one decision: who wakes up and why.
type SchedulerTick struct {
	When     time.Time
	Organism string // OrgArianna or OrgLeo
	Trigger  string // "cron" or "event:<reason>"
	PrevPrompt   string // last dialogue.response of the OTHER organism, if any
	PrevSpeaker  string // who said PrevPrompt (informational)
	PrevDebtDelta float64
	PrevChamber   string
}

// Run fires SchedulerTicks on the returned channel until ctx is done.
// Side-effect free except for the channel sends.
func Run(ctx context.Context, cfg SchedulerConfig) <-chan SchedulerTick {
	ch := make(chan SchedulerTick, 1)

	go func() {
		defer close(ch)

		// Initial alternation seed: who spoke last in KK? If empty, default
		// to Arianna (deepest field opens the conversation).
		next := OrgArianna
		if last, err := lastDialogueSpeaker(cfg.KKPath); err == nil && last != "" {
			next = otherOrganism(last) // listener becomes next speaker
		}

		ticker := time.NewTicker(cfg.CronInterval)
		defer ticker.Stop()

		emit := func(trigger string, why string) {
			t, _ := readLastDialogue(cfg.KKPath)
			tick := SchedulerTick{
				When:          time.Now().UTC(),
				Organism:      next,
				Trigger:       trigger + iff(why != "", ":"+why, ""),
				PrevPrompt:    t.Response,
				PrevSpeaker:   t.Speaker,
				PrevDebtDelta: t.ProphecyDebtDelta,
				PrevChamber:   t.DominantChamber,
			}
			select {
			case ch <- tick:
			case <-ctx.Done():
				return
			}
			next = otherOrganism(next)
		}

		// First tick fires immediately (the schedule is "alive now")
		emit("cron", "boot")

		// Event-check ticker: every 5 minutes, peek at KK for triggers.
		eventCheck := time.NewTicker(5 * time.Minute)
		defer eventCheck.Stop()

		for {
			select {
			case <-ctx.Done():
				return

			case <-ticker.C:
				emit("cron", "")

			case <-eventCheck.C:
				// Read latest dialogue row, decide if event-trigger is hot.
				t, err := readLastDialogue(cfg.KKPath)
				if err != nil {
					continue
				}
				if t.ProphecyDebtDelta > cfg.ProphecyDebtTrig {
					// Force wake Arianna if not already her turn.
					if next == OrgArianna {
						emit("event", fmt.Sprintf("prophecy_debt=%.2f", t.ProphecyDebtDelta))
						// reset cron alignment after early wake
						ticker.Reset(cfg.CronInterval)
					}
					continue
				}
				if cfg.VoidFlowTrig && (t.DominantChamber == "VOID" || t.DominantChamber == "FLOW") {
					if next == OrgLeo {
						emit("event", "chamber="+t.DominantChamber)
						ticker.Reset(cfg.CronInterval)
					}
				}
			}
		}
	}()

	return ch
}

// dialogueRow mirrors the KK dialogue table.
type dialogueRow struct {
	ID                int64
	TS                int64
	Speaker           string
	Listener          string
	Prompt            string
	Response          string
	ProphecyDebtDelta float64
	DominantChamber   string
}

// readLastDialogue shells out to sqlite3 CLI (Phase 3 MVP — no Go SQLite
// driver dep). Phase 5+ may swap this for modernc.org/sqlite when we add
// embedding-cosine retrieval.
func readLastDialogue(kkPath string) (dialogueRow, error) {
	var r dialogueRow

	// We use a uniquely improbable separator so prompt/response text
	// containing newlines or quotes doesn't break parsing.
	const SEP = "<<<NA__SEP>>>"
	q := fmt.Sprintf(
		"SELECT IFNULL(id,0)||'%s'||IFNULL(ts,0)||'%s'||IFNULL(speaker,'')||'%s'"+
			"||IFNULL(listener,'')||'%s'||IFNULL(prompt,'')||'%s'||IFNULL(response,'')||'%s'"+
			"||IFNULL(prophecy_debt_delta,0)||'%s'||IFNULL(dominant_chamber,'') "+
			"FROM dialogue ORDER BY ts DESC LIMIT 1;",
		SEP, SEP, SEP, SEP, SEP, SEP, SEP)

	out, err := exec.Command("sqlite3", kkPath, q).Output()
	if err != nil {
		return r, err
	}
	parts := strings.SplitN(strings.TrimRight(string(out), "\n"), SEP, 8)
	if len(parts) < 8 {
		return r, nil // empty dialogue table — leave zero values
	}
	r.ID, _ = strconv.ParseInt(parts[0], 10, 64)
	r.TS, _ = strconv.ParseInt(parts[1], 10, 64)
	r.Speaker = parts[2]
	r.Listener = parts[3]
	r.Prompt = parts[4]
	r.Response = parts[5]
	r.ProphecyDebtDelta, _ = strconv.ParseFloat(parts[6], 64)
	r.DominantChamber = parts[7]
	return r, nil
}

func lastDialogueSpeaker(kkPath string) (string, error) {
	r, err := readLastDialogue(kkPath)
	if err != nil {
		return "", err
	}
	return r.Speaker, nil
}

func otherOrganism(o string) string {
	if o == OrgArianna {
		return OrgLeo
	}
	return OrgArianna
}

// iff is the missing ternary
func iff(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}
