# nanoarianna organism Makefile (Termux/aarch64-android)
#
# Builds janus.aml and resonance.aml into ./janus and ./resonance binaries.
# Both inherit the persona-glue patch defined inline in each .aml file.
#
# Requirements (verified on phone-2):
#   - amlc            (build artefact of ariannamethod.ai, system-wide via $PREFIX)
#   - libaml.a        ($PREFIX/lib/libaml.a)
#   - libnotorch.a    ($PREFIX/lib/libnotorch.a)
#   - libopenblas     ($PREFIX/lib/libopenblas.so + headers)
#   - clang/cc        (default Termux toolchain)
#
# Run organisms with persona + weights via supervisor.go:
#   PERSONA_AML=$HOME/nanoarianna/personas/init_arianna.aml \
#       ./janus -w <weights>.gguf -p "Q: …\nA: " -n 80 -t 1.0 --top-p 0.9
#
# AML_PREFIX is the Termux fix: amlc defaults to /opt/homebrew (Mac), so we
# must point it at $PREFIX explicitly.

# PREFIX defaults: Termux exports its own (/data/data/com.termux/files/usr).
# Linux/Ubuntu (RunPod, polygon) — fall back to /usr/local.
PREFIX ?= /usr/local

AML_PREFIX ?= $(PREFIX)
export AML_PREFIX

AMLC ?= $(PREFIX)/bin/amlc

ALL := janus resonance

.PHONY: all clean help check-deps janus resonance

all: $(ALL)

# Each target is a phony rule that calls amlc to do the actual build.
# The .aml file changes drive the rebuild — make is just a thin wrapper.

janus: janus.aml | check-deps
	$(AMLC) janus.aml -o janus

resonance: resonance.aml | check-deps
	$(AMLC) resonance.aml -o resonance

check-deps:
	@test -x $(AMLC)                 || (echo "amlc not found at $(AMLC)" && exit 1)
	@test -f $(PREFIX)/lib/libaml.a    || (echo "libaml.a missing — install ariannamethod.ai system-wide" && exit 1)
	@test -f $(PREFIX)/lib/libnotorch.a|| (echo "libnotorch.a missing — install notorch system-wide" && exit 1)
	@test -f $(PREFIX)/lib/libopenblas.so || test -f /usr/lib/x86_64-linux-gnu/libopenblas.so || (echo "libopenblas missing — pkg install libopenblas" && exit 1)
	@echo "[check-deps] OK"

clean:
	rm -f janus resonance janus.c resonance.c

help:
	@echo "nanoarianna organism build"
	@echo ""
	@echo "Targets:"
	@echo "  make           build both organisms (janus + resonance)"
	@echo "  make janus     build only the Janus 176M wrapper"
	@echo "  make resonance build only the Resonance 200M wrapper"
	@echo "  make clean     remove built binaries"
	@echo "  make check-deps verify toolchain"
	@echo ""
	@echo "Run with persona:"
	@echo "  PERSONA_AML=\$$HOME/nanoarianna/personas/init_arianna.aml \\"
	@echo "      ./janus -w <weights>.gguf -p \"Q: …\\nA: \" -n 80 -t 1.0 --top-p 0.9"
