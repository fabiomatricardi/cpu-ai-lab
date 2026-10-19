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
    python tools/verify_flags.py --server             # LAUNCHER contract: chrome flags + presets from src/models.json
    python tools/verify_flags.py --server --dump      # save engine/llama-server-help.txt

Server mode also enforces decision policy (D27 no dupes, D30 no reasoning flags in
presets, D31 harness quant policy, D32 MTP scope, D33 escape needs reason, D34 port).
"""

import argparse
import json
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

# ---------------------------------------------------------------------------
# Server gate (llama-server — launcher contract), decision D22.
# Chrome flags are emitted by the launcher UI regardless of preset;
# preset flags come from src/models.json (single source of truth).
# ---------------------------------------------------------------------------
DEFAULT_ENGINE_SERVER = PROJECT_ROOT / "engine" / "vulkan" / "llama-server.exe"
DEFAULT_SERVER_HELP_DUMP = PROJECT_ROOT / "engine" / "llama-server-help.txt"
MODELS_JSON = PROJECT_ROOT / "src" / "models.json"

SERVER_CHROME_FLAGS = [
    ("-m",             "model path (launcher picker)"),
    ("-t",             "cpu threads (UI control)"),
    ("-c",             "context size (preset default, UI override)"),
    ("-np",            "parallel slots"),
    ("-ngl",           "gpu layers (Vulkan offload)"),
    ("-fa",            "flash attention"),
    ("-lv",            "log verbosity (memory breakdown)"),
    ("-ctk",           "KV cache quantization - K (control)"),
    ("-ctv",           "KV cache quantization - V (control)"),
    ("--jinja",        "chat template (always on)"),
    ("--host",         "bind address (always 127.0.0.1, D13)"),
    ("--port",         "port (default 11434, D34)"),
    ("--alias",        "model display name"),
    ("--ui",           "built-in web chat UI"),
    ("--ui-mcp-proxy", "MCP proxy for Exa web search (localhost only, D12)"),
    ("--reasoning-budget", "the ONLY reasoning flag the launcher emits (D30)"),
]

# D30: presets express reasoning via the `reasoning_default` field only —
# neither reasoning flag may ever live inside a preset array.
BANNED_PRESET_FLAGS = {"--reasoning-effort", "--reasoning-budget"}
# D32: speculative/MTP flags are only legal on presets with mtp == "in-gguf".
SPEC_PREFIXES = ("--spec-type", "--spec-draft-")


def extract_flags(flag_array) -> list:
    """Command-ready [flag, value, ...] array -> list of flag tokens.

    Value tokens that look like negative numbers ('-1' for --top-k) are not flags.
    """
    flags = []
    for tok in flag_array:
        tok = str(tok)
        if re.fullmatch(r"-\d+(\.\d+)?", tok):
            continue
        if tok.startswith("-") and len(tok) > 1:
            flags.append(tok)
    return flags


def server_gate(help_text: str, do_dump: bool) -> int:
    if do_dump:
        DEFAULT_SERVER_HELP_DUMP.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_SERVER_HELP_DUMP.write_text(help_text, encoding="utf-8")
        print(f"saved server help dump -> {DEFAULT_SERVER_HELP_DUMP}")

    present = parse_option_tokens(help_text)
    failures: list = []
    warnings: list = []

    print(f"CPU AI Lab flag gate [server] | pinned engine: {PINNED_BUILD}")
    print(f"parsed {len(present)} option tokens from llama-server help\n")

    print("-- launcher chrome flags --")
    for flag, purpose in SERVER_CHROME_FLAGS:
        ok = flag in present
        print(f"  [{'OK  ' if ok else 'FAIL'}] {flag:<24} {purpose}")
        if not ok:
            failures.append(flag)

    print("\n-- presets (src/models.json) --")
    if not MODELS_JSON.exists():
        failures.append("src/models.json not found")
        print(f"  [FAIL] {MODELS_JSON} not found")
    else:
        data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
        defaults = data.get("defaults", {})
        presets = data.get("presets", [])

        # D34 — default port
        if defaults.get("port") != 11434:
            failures.append("defaults.port != 11434 (D34)")
            print(f"  [FAIL] defaults.port = {defaults.get('port')!r} (D34 requires 11434)")
        else:
            print("  [OK  ] defaults.port = 11434 (D34)")

        # D31 — harness quant policy pinned
        harness = defaults.get("harness", {})
        if harness.get("preferred_quant") != "Q6_K" or harness.get("blocked_at_or_below") != "Q4_K_M":
            failures.append("harness quant policy drift (D31)")
            print(f"  [FAIL] harness = {harness} (D31: preferred Q6_K, blocked <= Q4_K_M)")
        else:
            print(f"  [OK  ] harness policy: preferred Q6_K | blocked <= Q4_K_M | KV {harness.get('kv_quant')} (D31)")

        if not presets:
            failures.append("presets list is empty")
        seen_ids = set()
        for p in presets:
            pid = str(p.get("id", "?"))
            if pid in seen_ids:
                failures.append(f"duplicate preset id: {pid}")
            seen_ids.add(pid)

            flags = extract_flags(p.get("flags", []))
            dupes = sorted({f for f in flags if flags.count(f) > 1})
            banned = sorted(f for f in flags if f in BANNED_PRESET_FLAGS)
            scoped = sorted(f for f in flags if f.startswith(SPEC_PREFIXES))
            missing = sorted({f for f in flags if f not in present})

            problems = []
            if missing:
                problems.append(f"not in engine: {', '.join(missing)}")
            if dupes:
                problems.append(f"duplicate flags (D27): {', '.join(dupes)}")
            if banned:
                problems.append(f"reasoning flag inside preset (D30 — use reasoning_default): {', '.join(banned)}")
            if scoped and p.get("mtp") != "in-gguf":
                problems.append(f"spec flags on non-MTP preset (D32): {', '.join(scoped)}")
            if p.get("agent_ok") and not p.get("agent_reason"):
                problems.append("agent_ok without agent_reason (D33)")

            meta = f"flags={len(flags):>2} | quant={p.get('quant')} | mtp={p.get('mtp')}"
            print(f"  [{'OK  ' if not problems else 'FAIL'}] {pid:<24} {meta}")
            for pb in problems:
                print(f"          -> {pb}")
            failures.extend(f"{pid}: {pb}" for pb in problems)

    print()
    for flag, note in REMOVED_FLAGS:
        if flag in present:
            warnings.append(f"{flag} present in server help — engine restored it ({note})")
            print(f"  [WARN] {flag:<24} {note}")
        else:
            print(f"  [GONE] {flag:<24} {note}  (as expected)")

    print()
    if failures:
        print(f"RESULT: FAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    for w in warnings:
        print(f"warning: {w}")
    print("RESULT: PASS")
    return 0


def parse_option_tokens(help_text: str) -> set:
    """Extract every option token from `--help` output.

    llama.cpp prints options like:
        -ctk,  --cache-type-k TYPE              KV cache data type for K
        -a,    --alias STRING                   set model name aliases...
    The option list may itself contain runs of 2+ spaces (short form -> long
    form), so the description boundary is: the first run of 2+ spaces whose
    following text does NOT start with '-'. Tokens are the comma-separated
    pieces of everything before that boundary. Continuation lines (indented)
    are ignored.
    """
    tokens: set = set()
    for line in help_text.splitlines():
        if not line or line[0] in " \t":
            continue  # continuation / description line
        opt_part = line
        for m in re.finditer(r"\s{2,}", line):
            after = line[m.end():]
            if after and not after.startswith("-"):
                opt_part = line[:m.start()]
                break
        for piece in opt_part.split(","):
            piece = piece.strip()
            # strip a trailing value placeholder: "N", "TYPE", "MODE", "FNAME"...
            tok = piece.split()[0] if piece else ""
            if tok.startswith("-") and len(tok) > 1:
                # ignore pure punctuation separators
                tokens.add(tok)
    return tokens


def get_help_text(help_file=None, binary: str = "llama-cli.exe") -> str:
    if help_file:
        return Path(help_file).read_text(encoding="utf-8", errors="replace")
    guess = Path(sys.argv[0]).resolve().parent.parent / "engine" / "vulkan" / binary
    exe = guess if guess.exists() else PROJECT_ROOT / "engine" / "vulkan" / binary
    if not exe.exists():
        print(f"FATAL: engine binary not found at {exe}")
        sys.exit(2)
    proc = subprocess.run(
        [str(exe), "-h"],
        capture_output=True, text=True, errors="replace", timeout=60,
    )
    return proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU AI Lab flag verification gate")
    ap.add_argument("--help-file", help="verify against a saved help dump instead of running the engine")
    ap.add_argument("--dump", action="store_true", help=f"save help dump to {DEFAULT_HELP_DUMP} (server: {DEFAULT_SERVER_HELP_DUMP.name})")
    ap.add_argument("--server", action="store_true",
                    help="verify llama-server contract: chrome flags + src/models.json presets + decision-policy checks")
    args = ap.parse_args()

    if args.server:
        help_text = get_help_text(args.help_file, "llama-server.exe")
        return server_gate(help_text, args.dump)

    help_text = get_help_text(args.help_file, "llama-cli.exe")
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
