#!/usr/bin/env python3
"""
CPU AI Lab - build-time flag verification gate.

Runs `llama-cli.exe -h` (or reads a saved help dump) and asserts that every
flag the app plans to pass actually exists in the pinned engine build.

This is the guard against the exact staleness that broke No Probl-LLAMA.cpp:
if an engine update removes/renames a flag, this script FAILS the build
before the app ships.

Usage:
    python tools/verify_flags.py                      # runs engine/vulkan/llama-cli.exe -h
    python tools/verify_flags.py --help-file path.txt # verify against a saved dump
    python tools/verify_flags.py --dump               # save the help dump next to the script
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PINNED_BUILD = "b11669"
PINNED_ZIP_SHA256 = "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e"
PINNED_ZIP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b11669/llama-b11669-bin-win-vulkan-x64.zip"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE_CLI = PROJECT_ROOT / "engine" / "vulkan" / "llama-cli.exe"
DEFAULT_HELP_DUMP = PROJECT_ROOT / "engine" / "llama-cli-help.txt"

# ---------------------------------------------------------------------------
# The contract: every flag the app emits, mapped to what it is used for.
# Each entry: (flag-as-typed-by-app, required?, purpose)
# ---------------------------------------------------------------------------
REQUIRED_FLAGS = [
    ("-m",            True,  "model path"),
    ("-t",            True,  "cpu threads"),
    ("-c",            True,  "context size"),
    ("-n",            True,  "n-predict (token count / 0 for inspect)"),
    ("-p",            True,  "prompt"),
    ("-f",            True,  "external prompt file"),
    ("-ngl",          True,  "gpu layers (Vulkan offload)"),
    ("-ctk",          True,  "KV cache quantization - K"),
    ("-ctv",          True,  "KV cache quantization - V"),
    ("-fa",           True,  "flash attention"),
    ("-st",           True,  "single-turn (clean exit for bench)"),
    ("-lv",           True,  "log verbosity (memory-breakdown parse)"),
    ("--no-display-prompt", True, "inspect mode prompt suppression"),
    ("--reasoning-budget",  True, "thinking-model budget control"),
    ("--list-devices",      True, "engine-level GPU/Vulkan device probe"),
    ("--version",           True, "engine build identity"),
]

# Flags we INTENTIONALLY do not use, with the replacement note (documentation
# gate: if one of these comes back, the engine restored old behavior).
REMOVED_FLAGS = [
    ("--mmap", "removed in 0.7.0 -> replaced by -lm/--load-mode "
               "(default 'auto' already mmaps; we pass nothing)"),
]

# New b11669 capabilities the paid suite will use (warn-only for Phase 1).
FUTURE_FLAGS = [
    ("--spec-draft-model",  "MTP / speculative draft model (-md)"),
    ("--spec-draft-ngl",    "draft model gpu layers (-ngld)"),
    ("--spec-draft-type-k", "draft KV cache quant K (-ctkd)"),
    ("--spec-draft-type-v", "draft KV cache quant V (-ctvd)"),
    ("--spec-type",         "speculative mode incl. draft-mtp"),
]


def parse_option_tokens(help_text: str) -> set:
    """Extract every option token from `--help` output.

    llama.cpp prints options like:
        -ctk,  --cache-type-k TYPE      KV cache data type for K
    Tokens are the comma-separated pieces before the first run of 2+ spaces.
    Continuation lines (indented) are ignored.
    """
    tokens: set = set()
    for line in help_text.splitlines():
        if not line or line[0] in " \t":
            continue  # continuation / description line
        head = re.split(r"\s{2,}", line, maxsplit=1)[0]
        for piece in head.split(","):
            piece = piece.strip()
            # strip a trailing value placeholder: "N", "TYPE", "MODE", "FNAME"...
            tok = piece.split()[0] if piece else ""
            if tok.startswith("-") and len(tok) > 1:
                # ignore pure punctuation separators
                tokens.add(tok)
    return tokens


def get_help_text(help_file=None) -> str:
    if help_file:
        return Path(help_file).read_text(encoding="utf-8", errors="replace")
    cli = Path(sys.argv[0]).resolve().parent.parent / "engine" / "vulkan" / "llama-cli.exe"
    if not cli.exists():
        cli = DEFAULT_ENGINE_CLI
    if not cli.exists():
        print(f"FATAL: engine CLI not found at {cli}")
        sys.exit(2)
    proc = subprocess.run(
        [str(cli), "-h"],
        capture_output=True, text=True, errors="replace", timeout=60,
    )
    return proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU AI Lab flag verification gate")
    ap.add_argument("--help-file", help="verify against a saved help dump instead of running the engine")
    ap.add_argument("--dump", action="store_true", help=f"save help dump to {DEFAULT_HELP_DUMP}")
    args = ap.parse_args()

    help_text = get_help_text(args.help_file)
    if args.dump:
        DEFAULT_HELP_DUMP.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_HELP_DUMP.write_text(help_text, encoding="utf-8")
        print(f"saved help dump -> {DEFAULT_HELP_DUMP}")

    present = parse_option_tokens(help_text)
    failures = []
    warnings = []

    print(f"CPU AI Lab flag gate | pinned engine: {PINNED_BUILD}")
    print(f"parsed {len(present)} option tokens from help output\n")

    for flag, required, purpose in REQUIRED_FLAGS:
        ok = flag in present
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {flag:<24} {purpose}")
        if not ok and required:
            failures.append(flag)

    print()
    for flag, note in REMOVED_FLAGS:
        if flag in present:
            warnings.append(f"{flag} is present again — engine restored it; reconsider ({note})")
            print(f"  [WARN] {flag:<24} {note}")
        else:
            print(f"  [GONE] {flag:<24} {note}  (as expected)")

    print()
    for flag, purpose in FUTURE_FLAGS:
        if flag in present:
            print(f"  [FUTR] {flag:<24} {purpose}")
        else:
            warnings.append(f"{flag} missing — paid-suite feature unavailable on this engine ({purpose})")
            print(f"  [MISS] {flag:<24} {purpose}  <-- not in this build")

    print()
    if failures:
        print(f"RESULT: FAIL — {len(failures)} required flag(s) missing: {', '.join(failures)}")
        return 1
    for w in warnings:
        print(f"warning: {w}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
