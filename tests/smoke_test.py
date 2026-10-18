#!/usr/bin/env python3
"""CPU AI Lab - end-to-end smoke test (headless-ish).

Creates the real GUI (withdrawn), lets engine setup + Vulkan detection run,
executes a real inference through build_cmd(), feeds every log line through
parse_line(), and asserts the dashboard + result card populate.

Run:  python tests/smoke_test.py
"""

import sys
import subprocess
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cpulab_bench as cb  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main():
    root = tk.Tk()
    root.withdraw()
    app = cb.CpuLabBench(root)

    # wait for async engine setup AND the end of Vulkan detection (the engine
    # status line is set before detection runs — must wait for gpu_status too)
    for _ in range(120):
        root.update()
        if (app.cli_path is not None
                and app.engine_status.get().startswith("Engine: llama.cpp")
                and not app.gpu_status.get().startswith("GPU: detecting")):
            break
        time.sleep(0.25)
    root.update()

    print("== setup ==")
    check("engine found", app.cli_path is not None, str(app.cli_path))
    check("engine status pinned build", "b11669" in app.engine_status.get(),
          app.engine_status.get())
    check("vulkan detected", app.vulkan_ok, str(app.vulkan_device))
    check("ngl defaults to 99 on Vulkan", app.ngl.get() == 99)
    root.update()

    model = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"
    check("test model present", model.exists(), str(model))
    app.model_path.set(str(model))

    print("== layout ==")

    def walk(w):
        for c in w.winfo_children():
            yield c
            yield from walk(c)

    # v0.1.0 bug regression: "Get test model" was gridded into the SAME cell as
    # the .gguf "Browse" picker and completely covered it. Two widgets must never
    # share one grid cell (per container) unless intentionally stacked.
    dups = []
    for parent in [root] + list(walk(root)):
        cells = {}
        for ch in parent.winfo_children():
            gi = ch.grid_info()
            if not gi or "row" not in gi:
                continue  # pack-managed, not applicable
            cells.setdefault((gi["row"], gi["column"]), []).append(ch)
        for key, kids in cells.items():
            if len(kids) > 1:
                desc = []
                for k in kids:
                    name = type(k).__name__
                    try:
                        name += "(" + str(k.cget("text")) + ")"
                    except tk.TclError:
                        pass
                    desc.append(name)
                dups.append(f"{type(parent).__name__}@{key}: {' + '.join(desc)}")
    check("no two widgets share a grid cell", not dups, "; ".join(dups))

    row1_btns = {}
    for ch in root.winfo_children():
        gi = ch.grid_info()
        if gi.get("row") == 1 and isinstance(ch, tk.Button):
            row1_btns[str(ch.cget("text"))] = gi["column"]
    check("model 'Browse' picker visible in column 2",
          row1_btns.get("Browse") == 2, str(row1_btns))
    check("'Get test model' in column 3 (not stacked on Browse)",
          row1_btns.get("Get test model") == 3, str(row1_btns))

    print("== command contract ==")
    run_cmd = app.build_cmd("run")
    insp_cmd = app.build_cmd("inspect")
    flat = " ".join(run_cmd)
    check("--mmap absent", "--mmap" not in flat)
    check("-lv 4 present", "-lv 4" in flat)
    check("-ngl 99 on vulkan", "-ngl 99" in flat)
    check("inspect uses -n 0 + --no-display-prompt",
          "-n 0" in " ".join(insp_cmd) and "--no-display-prompt" in " ".join(insp_cmd))
    check("prompt passed raw (no literal quotes)",
          '"Explain' not in flat and "Explain the difference" in flat)

    # no-Vulkan fallback contract
    saved = app.vulkan_ok
    app.vulkan_ok = False
    cpu_flat = " ".join(app.build_cmd("run"))
    check("-ngl 0 when no vulkan", "-ngl 0" in cpu_flat)
    app.vulkan_ok = saved

    print("== real inference run ==")
    proc = subprocess.run(run_cmd, capture_output=True, text=True,
                          errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    check("engine exit 0", proc.returncode == 0, f"rc={proc.returncode}")
    for line in out.splitlines():
        app.parse_line(line)
    root.update()

    check("layers parsed", app.stat_layers.get() not in ("Layers: -", ""),
          app.stat_layers.get())
    check("offload parsed", "/25 layers" in app.stat_offload.get() or
          "layers -> GPU" in app.stat_offload.get(), app.stat_offload.get())
    check("vram parsed", "MiB" in app.stat_vram.get(), app.stat_vram.get())
    check("prompt t/s parsed", "t/s" in app.stat_prompt_speed.get() and
          "-" != app.stat_prompt_speed.get().split()[-1], app.stat_prompt_speed.get())
    check("gen t/s parsed", "t/s" in app.stat_gen_speed.get() and
          "-" != app.stat_gen_speed.get().split()[-1], app.stat_gen_speed.get())
    check("verdict set", "Verdict:" in app.verdict_var.get() and
          "-" != app.verdict_var.get().split()[-1], app.verdict_var.get())
    check("weights (Host row) parsed", "Weights: -" != app.stat_model_ram.get(),
          app.stat_model_ram.get())
    check("result card dict complete",
          bool(app.last_result) and all(
              k in app.last_result for k in ("prompt_tps", "gen_tps", "verdict")),
          str(app.last_result))

    # result card text builds without error (dry-run the composition)
    card_bits = [app.stat_prompt_speed.get(), app.stat_gen_speed.get(),
                 app.verdict_var.get(), app.offload_txt or "-",
                 app.vram_txt or "-", cb.PINNED_BUILD]
    check("result card fields", all(card_bits), " | ".join(str(b) for b in card_bits))

    root.destroy()
    print()
    if FAILURES:
        print(f"SMOKE TEST: FAIL ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("SMOKE TEST: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
