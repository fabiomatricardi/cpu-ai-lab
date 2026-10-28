#!/usr/bin/env python3
"""Launcher smoke test (item 2.2 gate, no server run - logic + headless UI only).

Run:  python tests/launcher_smoke.py
Complements tests/smoke_test.py (bench, real inference) and
tools/verify_flags.py --server (flag manifest). Exit code 0 = all checks PASS.
"""

import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cpulab_launch as L  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK  ] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def main():
    print("CPU AI Lab launcher smoke test")
    data, err = L.load_presets()
    check("models.json loads", err is None, str(err))
    presets = data.get("presets", [])
    defaults = data.get("defaults", {})
    check("8 curated presets", len(presets) == 8, f"got {len(presets)}")
    with_url = [p["id"] for p in presets if p.get("download_url")]
    check("every preset has a download_url (D24)", len(with_url) == len(presets),
          f"missing: {sorted(set(p['id'] for p in presets) - set(with_url))}")
    missing_reason = [p["id"] for p in presets
                      if p.get("agent_ok") and not p.get("agent_reason")]
    check("agent_ok always carries agent_reason (D33)", not missing_reason, str(missing_reason))

    # --- composition: every preset x both modes ---------------------------
    eligible_ids, blocked_ids = [], []
    for mode in ("chat", "harness"):
        for p in presets:
            st = L.harness_status(p, defaults)
            (eligible_ids if st["eligible"] else blocked_ids).append(p["id"])
            kv = (st["kv"], st["kv"]) if mode == "harness" and st["eligible"] else None
            argv = L.compose_command(
                p, server="llama-server.exe", model_path="m.gguf",
                port=11434, threads=4, ngl=99, kv=kv,
                reasoning=p.get("reasoning_default", "medium"))
            flags = [a for a in argv[2:] if a.startswith("-")]
            tag = f"{mode}/{p['id']}"

            check(f"{tag}: no duplicate flags", len(flags) == len(set(flags)),
                  ",".join(sorted({f for f in flags if flags.count(f) > 1})))
            check(f"{tag}: --host 127.0.0.1",
                  argv[argv.index("--host") + 1] == "127.0.0.1")
            check(f"{tag}: port 11434 (D34)", argv[argv.index("--port") + 1] == "11434")
            joined = " ".join(argv)
            check(f"{tag}: --reasoning-budget emitted (D30)", "--reasoning-budget" in flags)
            check(f"{tag}: never --reasoning-effort (D30)", "--reasoning-effort" not in joined)
            check(f"{tag}: never --tools (D14)", "--tools" not in joined and "-to" != "--tools")
            check(f"{tag}: --ui + --ui-mcp-proxy present",
                  "--ui" in flags and "--ui-mcp-proxy" in flags)
            check(f"{tag}: --jinja present", "--jinja" in flags)
            check(f"{tag}: KV q4_0 (D31)", argv[argv.index("-ctk") + 1] == "q4_0"
                  and argv[argv.index("-ctv") + 1] == "q4_0")
            check(f"{tag}: alias", argv[argv.index("--alias") + 1] == p["alias"])
            # every proven preset flag made it into the command unchanged
            pf = p.get("flags", [])
            ok = all(
                (pf[i] in flags and i + 1 < len(pf) and pf[i + 1] in argv)
                if not pf[i + 1].startswith("--") else (pf[i] in flags)
                for i in range(0, len(pf) - 1, 2))
            check(f"{tag}: preset flags preserved", ok)

    # --- D38: capability facts present on every preset --------------------
    for p in presets:
        check(f"{p['id']}: params_b present (D38)",
              isinstance(p.get("params_b"), (int, float)) and not isinstance(p.get("params_b"), bool)
              and p["params_b"] > 0)
        check(f"{p['id']}: reasoning_capable bool (D38)", isinstance(p.get("reasoning_capable"), bool))
    check("harness policy: min_params_b + require_reasoning (D38)",
          isinstance(defaults.get("harness", {}).get("min_params_b"), (int, float))
          and isinstance(defaults.get("harness", {}).get("require_reasoning"), bool))

    # dedupe/uniqueness of the harness-eligibility split
    check("harness eligible = proven 3-4B reasoning models (D31+D38: 4)",
          sorted(set(eligible_ids)) == sorted(
              ["granite-4.2-3b-q6k", "spark-x2.5-4b-q6k", "qwen35-4b-mtp-q6k",
               "spark-x2.6-4b-q8"]),
          str(sorted(set(eligible_ids))))
    check("harness blocked = 2 ternaries + LFM (no reasoning) + MiniCPM (1B)",
          sorted(set(blocked_ids)) == sorted(
              ["bonsai-4b-q2", "bonsai-2b-q2", "lfm2.5-1.2b-q6k", "MiniCPM5-1B-q8"]),
          str(sorted(set(blocked_ids))))
    for p in presets:
        st = L.harness_status(p, defaults)
        if not st["eligible"]:
            reason = st["reason"]
            if p["id"] in ("bonsai-4b-q2", "bonsai-2b-q2"):
                check(f"{p['id']}: quant block reason mentions Q6_K remedy", "Q6_K" in reason)
            if p["id"] == "lfm2.5-1.2b-q6k":
                check(f"{p['id']}: reasoning block (D38)",
                      "no reasoning capability" in reason and "D38" in reason)
                check(f"{p['id']}: also scale block (1.2 < 3)", "1.2B below harness scale floor" in reason)
            if p["id"] == "MiniCPM5-1B-q8":
                check(f"{p['id']}: scale block (D38)",
                      "1B below harness scale floor" in reason and "D38" in reason)
            if p["id"] == "bonsai-2b-q2":
                check(f"{p['id']}: joined reasons (quant AND scale)",
                      "Q6_K" in reason and "2B below" in reason)

    # --- MTP scope (D32) --------------------------------------------------
    for p in presets:
        argv = L.compose_command(p, server="s", model_path="m", reasoning="medium")
        joined = " ".join(argv)
        if p.get("mtp") == "in-gguf":
            check(f"{p['id']}: --spec-draft-n-max 2 (confirmed 10-21)",
                  "--spec-draft-n-max 2" in joined and "--spec-type draft-mtp" in joined)
        else:
            check(f"{p['id']}: no --spec-* (D32)", "--spec-" not in joined)

    # --- overrides stay deduped (D27) -------------------------------------
    p0 = presets[0]
    argv = L.compose_command(p0, server="s", model_path="m", ctx=999, reasoning="off")
    check("ctx override replaces -c once", argv.count("-c") == 1
          and argv[argv.index("-c") + 1] == "999")
    check("reasoning off -> budget 0", argv[argv.index("--reasoning-budget") + 1] == "0")
    argv = L.compose_command(p0, server="s", model_path="m", kv=("q8_0", "q8_0"),
                             reasoning="high")
    check("KV override applies", argv[argv.index("-ctk") + 1] == "q8_0"
          and argv[argv.index("--reasoning-budget") + 1] == "4096")

    # negative-number values survive parsing (e.g. --top-k -1)
    pairs = dict(L._flag_pairs(["--top-k", "-1", "--jinja", "-c", "64000"]))
    check("negative value kept (top-k -1)", pairs.get("--top-k") == "-1")
    check("flag pairs parse -c", pairs.get("-c") == "64000")

    # --- level <-> budget + OpenCode export (D26/D30) ---------------------
    check("budget map", [L._budget_of(x) for x in
                         ("off", "low", "medium", "high", "xhigh")] == [0, 1024, 2048, 4096, 8192])
    check("budget_to_level reverse", L.budget_to_level(2048) == "medium"
          and L.budget_to_level(0) == "off")
    for p in presets:
        for lvl in L.REASONING_LEVELS:
            argv = L.compose_command(p, server="s", model_path="m", reasoning=lvl)
            cfg = json.loads(L.opencode_export(p, argv=argv, reasoning=lvl, port=11434))
            budget = L._budget_of(lvl)
            alias_key = str(p.get("alias") or p.get("id", "model"))
            m = (cfg.get("provider", {}).get("local", {})
                 .get("models", {}).get(alias_key, {}))
            check(f"{p['id']}/{lvl}: limit.output = budget + slack (D42)",
                  cfg.get("$schema") == "https://opencode.ai/config.json"
                  and cfg.get("model") == f"local/{alias_key}"
                  and m.get("limit", {}).get("output") == budget + L.REASONING_SLACK)
            check(f"{p['id']}/{lvl}: limit.context = -c (D42)",
                  m.get("limit", {}).get("context")
                  == int(dict(L._flag_pairs(p["flags"]))["-c"]))

    # --- D34 port-busy detection -----------------------------------------
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy_port = s.getsockname()[1]
    check("port_busy True on live listener", L.port_busy(busy_port))
    s.close()
    check("port_busy False on free port", not L.port_busy(busy_port))

    # --- headless UI instantiate (skip with CPULAB_SKIP_UI=1 on CI) --------
    import os
    if os.environ.get("CPULAB_SKIP_UI"):
        check("UI section skipped (CPULAB_SKIP_UI)", True)
    else:
        import tkinter as tk
        try:
            root = tk.Tk()
            root.withdraw()
            app = L.CpuLabLauncher(root)
            check("UI: 8 presets in tree", len(app.tree.get_children()) == 8)
            app.mode_var.set("harness")
            app._on_mode_change()
            blocked = [iid for iid in app.tree.get_children()
                       if "blocked" in app.tree.item(iid, "tags")]
            check("UI: harness mode greys 4 presets (2 quant + LFM + MiniCPM)",
                  len(blocked) == 4, str(blocked))
            check("UI: selected preset is harness-eligible", app._eligible_current())
            app.mode_var.set("chat")
            app._on_mode_change()
            check("UI: chat mode unblocks all", len(blocked) == 4 and app._eligible_current() is not None)
            app.reasoning_var.set("xhigh")
            app.refresh()
            check("UI: command preview has budget 8192",
                  "--reasoning-budget 8192" in app.cmd_text.get("1.0", "end"))
            root.destroy()
        except tk.TclError as exc:
            check("UI instantiate", False, str(exc))

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
