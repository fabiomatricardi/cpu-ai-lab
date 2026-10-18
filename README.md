# CPU AI Lab — Honest Bench

**A free, honest local-LLM benchmarker for Windows x64.**
Answers one question: *how fast can MY PC really run local AI?* — measured on your hardware, not on someone else's chart.

By **Fabio Matricardi — The Poor GPU Guy** · part of the **CPU AI Lab** line (*Bench → Run → Learn*).

## Features

- **Pinned engine** — on first run, downloads the exact llama.cpp build the app was tested against (**b11669**, win-x64 Vulkan zip), SHA256-verified. No "latest build" roulette.
- **Vulkan auto-detect → `-ngl` gate**, three layers deep:
  1. `vulkan-1.dll` present?
  2. `llama-cli.exe --list-devices` reports a `Vulkan` device? (authoritative)
  3. Run logs confirm `offloaded N/N layers to GPU`

  GPU confirmed → `-ngl 99` (adjustable) · otherwise → `-ngl 0`, silent CPU-only mode. You never type a flag.
- **Honest KPIs** — prompt (prefill) t/s, generation t/s, live RAM bar (Host memory row), VRAM breakdown, layer offload, plus a plain-language verdict: SNAPPY / USABLE / SLOW / TOO SLOW.
- **First-run warm-up hint** — the first GPU run compiles shaders; run twice, trust the second number.
- **Get test model** — one click, tiered by your installed RAM (≥12 GB high tier, ≥9 GB mid, else low).
- **Copy Result Card** — compact text block with hardware, engine build, offload, VRAM and t/s — ready to paste in a forum or GitHub issue.

## Quick start (user)

1. Download `CPU_AI_Lab_Bench.exe` from [Releases](../../releases).
2. Double-click. First run downloads the pinned engine (~31 MB) next to the exe.
3. Press **Get test model** → **Inspect** → **Run**.
4. Hit **Copy Result Card** and share your numbers!

## Quick start (developer)

```bash
git clone https://github.com/fabiomatricardi/cpu-ai-lab.git
cd cpu-ai-lab
pip install -r requirements.txt

# provision the pinned engine (dev layout expected by the tools)
mkdir -p engine/vulkan
curl -L -o engine/llama-b11669-bin-win-vulkan-x64.zip \
  https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-win-vulkan-x64.zip
# expected SHA256: 2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e
# unzip it into engine/vulkan/ so that engine/vulkan/llama-cli.exe exists

python src/cpulab_bench.py
```

## The flag gate — run before every build

```bash
python tools/verify_flags.py                        # against the local engine
python tools/verify_flags.py --help-file docs/llama-cli-b11669-help.txt   # against the committed snapshot
```

Parses `llama-cli.exe -h` and **fails (exit 1)** if any flag the app uses has been removed or renamed by an engine update. It also tracks flags that are *supposed* to be gone (`--mmap`) and new capabilities (`--spec-draft-*`). CI runs the snapshot variant on every push ([`.github/workflows/flag-gate.yml`](.github/workflows/flag-gate.yml)).

**This is the guard against silent engine churn.** Example: llama.cpp 0.7.0 removed `--mmap` (replaced by `--load-mode`, default `auto` already mmaps) and moved the memory-breakdown log lines to verbosity level 4 — this repo was rebuilt for that reality.

## Engine contract (pinned)

| Item | Value |
|---|---|
| Build | llama.cpp **b11669** (0.7.0-dev, commit 620d2225c) |
| Zip | `llama-b11669-bin-win-vulkan-x64.zip` |
| SHA256 | `2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e` |
| Log level | `-lv 4` (trace — required for memory breakdown / `n_layer` / offload lines) |
| Removed flag | `--mmap` → load-mode `auto`; the app passes nothing |
| New in 0.7.0 | `--spec-draft-model/-md`, `--spec-draft-ngl/-ngld`, `-ctkd/-ctvd`, `--spec-type …draft-mtp…` |

The full help snapshot of the pinned build is committed at [`docs/llama-cli-b11669-help.txt`](docs/llama-cli-b11669-help.txt).

## Test

```bash
python tools/verify_flags.py     # flag gate — must PASS
python tests/smoke_test.py       # engine setup → real inference → parser assertions — must ALL PASS
```

The smoke test runs a real inference pass against a GGUF in `models/` (use **Get test model** in the app first, or drop any small Q4_K_M GGUF there).

## Build the release exe

```bash
python tools/verify_flags.py
python tests/smoke_test.py
pyinstaller --noconsole --onefile --name "CPU_AI_Lab_Bench" src/cpulab_bench.py
```

The frozen exe writes `engine/` and `models/` **next to itself** (not in the PyInstaller temp dir).

## Roadmap

- **Now:** free benchmarker — measure honestly.
- **Next:** "What can I run?" recommender (your hardware → ranked models with realistic expected speeds) and a one-click launcher for the new llama.cpp flag era (MTP draft models, KV-cache quantization presets) — all against the same pinned-engine + flag-gate discipline.

## Credits & License

Engine: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT/AGPPT components — see the engine zip), pinned build b11669.
Bench UI lineage: Fabio's [tkinter-llamaCLI-speedTest](https://github.com/fabiomatricardi/tkinter-llamaCLI-speedTest).

MIT License — Copyright (c) 2026 **Fabio Matricardi — The Poor GPU Guy**. See [LICENSE](LICENSE).
