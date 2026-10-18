#!/usr/bin/env python3
"""
CPU AI Lab - Honest Bench  (Phase 1: free benchmarker)

Answers one question honestly: "How fast can MY PC really run local AI?"

Ported from TK_GGUF_bench_v1.6.5.py, reworked for:
  - pinned llama.cpp engine (b11669, win-x64 Vulkan zip, SHA256-verified)
  - Vulkan auto-detection -> enables -ngl ONLY when the engine confirms a
    Vulkan device, otherwise silent CPU-only mode
  - --mmap flag removed (gone in llama.cpp 0.7.0; load-mode auto = mmap)
  - auto engine download on first run (no terminal, no zip juggling)
  - one-click tiered test-model download (for first-time users)
  - shareable result card + honest speed verdict
  - teaser for the paid "What can I run?" recommender

Build:  pyinstaller --noconsole --onefile --name "CPU_AI_Lab_Bench" src/cpulab_bench.py
Gate:   python tools/verify_flags.py   (must PASS before shipping)
"""

import hashlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import psutil

APP_NAME = "CPU AI Lab - Honest Bench"
APP_VERSION = "0.1.0"

# --- Pinned engine (Phase 1 contract) -------------------------------------
PINNED_BUILD = "b11669"
PINNED_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b11669/llama-b11669-bin-win-vulkan-x64.zip"
)
PINNED_SHA256 = "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e"

# Test models by RAM tier (same stacking logic as GGUF-Ecosystem-Installer)
LOW_TIER = [
    ("Qwen3.5-0.8B (Q4_K_M)",
     "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf?download=true"),
]
MID_TIER = [
    ("Qwen3.5-2B (Q4_K_M)",
     "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true"),
    ("Llama-3.2-3B (Q4_K_M)",
     "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"),
]
HIGH_TIER = [
    ("Gemma-3n-E2B (Q4_K_M)",
     "https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf?download=true"),
    ("Qwen3-4B-Instruct (Q4_K_S)",
     "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_S.gguf?download=true"),
]


def app_dir() -> Path:
    """Folder for user data (engine/, models/): next to the .exe when frozen,
    otherwise the source folder. NOT the PyInstaller temp dir."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ENGINE_DIR = app_dir() / "engine"
MODELS_DIR = app_dir() / "models"


def find_cli() -> Path | None:
    """Locate llama-cli.exe (flat install layout or dev engine/vulkan layout)."""
    for cand in (ENGINE_DIR / "llama-cli.exe",
                 app_dir() / "engine" / "vulkan" / "llama-cli.exe",
                 Path(__file__).resolve().parent.parent / "engine" / "vulkan" / "llama-cli.exe"):
        if cand.exists():
            return cand
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_vulkan(cli: Path):
    """Vulkan detection, layered:
    1) loader DLL present?
    2) engine itself lists a Vulkan device? (authoritative)
    Returns (dll_found, device_line or None).
    """
    dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll"
    dll_found = dll.exists()
    if not dll_found:
        return False, None
    try:
        proc = subprocess.run(
            [str(cli), "--list-devices"],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        for line in out.splitlines():
            if "Vulkan" in line:
                return True, line.strip()
    except Exception:
        pass
    return True, None  # loader present but engine probe failed -> caller decides


def engine_version(cli: Path) -> str:
    try:
        proc = subprocess.run([str(cli), "--version"],
                              capture_output=True, text=True, errors="replace", timeout=30)
        first = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        return first[0] if first else "unknown"
    except Exception:
        return "unknown"


class CpuLabBench:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("950x680")
        self.process = None
        self.is_stopping = False
        self.last_result = None  # for the result card

        # Thread-safe UI updates: workers enqueue closures, main thread drains.
        self.ui_queue = queue.Queue()
        root.after(80, self._pump_ui)

        mem = psutil.virtual_memory()
        self.total_sys_ram_gb = mem.total / (1024 ** 3)
        self.max_threads = psutil.cpu_count(logical=True) or 4
        default_threads = max(1, self.max_threads // 2)

        self.cli_path = find_cli()
        self.vulkan_ok = False
        self.vulkan_device = None

        # ---------------- Top: engine / GPU status ----------------
        top = tk.Frame(root)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 0))
        self.engine_status = tk.StringVar(value="Engine: checking...")
        self.gpu_status = tk.StringVar(value="GPU: detecting...")
        tk.Label(top, textvariable=self.engine_status, fg="grey35",
                 font=("Arial", 9)).pack(side="left")
        tk.Label(top, textvariable=self.gpu_status, fg="#0a66c2",
                 font=("Arial", 9, "bold")).pack(side="right")

        # ---------------- 1. Model selection ----------------
        tk.Label(root, text="Model File (GGUF):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.model_path = tk.StringVar()
        tk.Entry(root, textvariable=self.model_path, width=62).grid(row=1, column=1, padx=5)
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=1, column=2, padx=5)
        self.get_model_btn = tk.Button(root, text="Get test model", command=self.download_test_model,
                                       bg="#f0ad4e", fg="black")
        self.get_model_btn.grid(row=1, column=2, padx=5, sticky="e")
        root.grid_columnconfigure(1, weight=1)

        # ---------------- 2. Prompt file ----------------
        tk.Label(root, text="Prompt File (.txt):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.prompt_file_path = tk.StringVar()
        tk.Entry(root, textvariable=self.prompt_file_path, width=62).grid(row=2, column=1, padx=5)
        tk.Button(root, text="Browse", command=self.browse_prompt_file).grid(row=2, column=2, padx=5)

        # ---------------- 3. Custom prompt ----------------
        tk.Label(root, text="Custom Prompt:").grid(row=3, column=0, sticky="nw", padx=10, pady=5)
        self.custom_prompt = tk.Text(root, height=3, width=75, font=("Arial", 9))
        self.custom_prompt.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.custom_prompt.insert(
            "1.0", "Explain the difference between CPU and GPU inference in simple terms.")

        # ---------------- 4. Parameters ----------------
        param_frame = tk.Frame(root)
        param_frame.grid(row=4, column=0, columnspan=3, sticky="w", padx=10)

        tk.Label(param_frame, text="GPU Layers (-ngl):").pack(side="left")
        self.ngl = tk.IntVar(value=0)
        self.ngl_spin = tk.Spinbox(param_frame, from_=0, to=99, textvariable=self.ngl, width=5)
        self.ngl_spin.pack(side="left", padx=5)

        tk.Label(param_frame, text=f"Threads (Max {self.max_threads}):").pack(side="left", padx=5)
        self.threads = tk.IntVar(value=default_threads)
        tk.Spinbox(param_frame, from_=1, to=self.max_threads, textvariable=self.threads,
                   width=5).pack(side="left", padx=5)

        tk.Label(param_frame, text="Context:").pack(side="left", padx=5)
        self.context = tk.IntVar(value=4096)
        tk.Spinbox(param_frame, from_=2048, to=128000, textvariable=self.context,
                   width=8).pack(side="left", padx=5)

        tk.Label(param_frame, text="Tokens (-n):").pack(side="left", padx=5)
        self.n_predict = tk.IntVar(value=200)
        tk.Spinbox(param_frame, from_=0, to=20000, textvariable=self.n_predict,
                   width=7).pack(side="left", padx=5)

        tk.Label(param_frame, text="Budget (-1/0/1..8192):").pack(side="left", padx=5)
        self.reasoning_budget = tk.IntVar(value=0)
        tk.Spinbox(param_frame, from_=-1, to=8192, textvariable=self.reasoning_budget,
                   width=6).pack(side="left", padx=5)

        cb_frame = tk.Frame(root)
        cb_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=10)
        self.ctk_var, self.ctv_var, self.fa_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        tk.Checkbutton(cb_frame, text="Quantize K Cache (q8_0)", variable=self.ctk_var).pack(side="left", padx=5)
        tk.Checkbutton(cb_frame, text="Quantize V Cache (q8_0)", variable=self.ctv_var).pack(side="left", padx=5)
        tk.Checkbutton(cb_frame, text="Flash Attention", variable=self.fa_var).pack(side="left", padx=5)

        # ---------------- 5. Stats panel ----------------
        info_frame = tk.LabelFrame(root, text=" Live Model Stats & Performance ",
                                   fg="#0a66c2", font=("Arial", 10, "bold"))
        info_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=8)

        self.stat_layers = tk.StringVar(value="Layers: -")
        self.stat_offload = tk.StringVar(value="Offload: -")
        self.stat_vram = tk.StringVar(value="VRAM: -")
        self.stat_free_ram = tk.StringVar(value=f"System Available: {mem.available / (1024 ** 3):.2f} GB")
        self.stat_model_ram = tk.StringVar(value="Weights: -")
        self.stat_kv_ram = tk.StringVar(value="KV Cache: -")
        self.stat_buffer_ram = tk.StringVar(value="Compute Buffer: -")
        self.stat_total_ram = tk.StringVar(value="Model Est. Total: -")
        self.stat_prompt_speed = tk.StringVar(value="Prompt: -")
        self.stat_gen_speed = tk.StringVar(value="Gen: -")

        tk.Label(info_frame, textvariable=self.stat_layers).grid(row=0, column=0, padx=20, sticky="w")
        tk.Label(info_frame, textvariable=self.stat_free_ram, fg="green").grid(row=0, column=1, padx=20, sticky="w")
        tk.Label(info_frame, textvariable=self.stat_offload, fg="#0a66c2").grid(row=0, column=2, padx=20, sticky="w")
        tk.Label(info_frame, textvariable=self.stat_model_ram).grid(row=1, column=0, padx=20, sticky="w")
        tk.Label(info_frame, textvariable=self.stat_kv_ram).grid(row=1, column=1, padx=20, sticky="w")
        tk.Label(info_frame, textvariable=self.stat_vram, fg="#7b2fbf").grid(row=1, column=2, padx=20, sticky="w")
        tk.Label(info_frame, textvariable=self.stat_buffer_ram).grid(row=2, column=0, padx=20, sticky="w")
        self.ram_display = tk.Label(info_frame, textvariable=self.stat_total_ram,
                                    font=("Arial", 9, "bold"))
        self.ram_display.grid(row=2, column=1, padx=20, sticky="w")

        perf = tk.Frame(info_frame)
        perf.grid(row=3, column=0, columnspan=2, pady=5, sticky="w")
        tk.Label(perf, textvariable=self.stat_prompt_speed, fg="purple",
                 font=("Arial", 9, "bold")).pack(side="left", padx=20)
        tk.Label(perf, textvariable=self.stat_gen_speed, fg="darkred",
                 font=("Arial", 9, "bold")).pack(side="left", padx=20)
        self.verdict_var = tk.StringVar(value="Verdict: -")
        tk.Label(perf, textvariable=self.verdict_var, fg="#0a66c2",
                 font=("Arial", 9, "bold")).pack(side="left", padx=20)

        self.canvas_width = 840
        self.canvas = tk.Canvas(info_frame, width=self.canvas_width, height=25,
                                bg="#e0e0e0", highlightthickness=1)
        self.canvas.grid(row=4, column=0, columnspan=2, padx=20, pady=5)

        legend = tk.Frame(info_frame)
        legend.grid(row=5, column=0, columnspan=2, pady=2)
        tk.Label(legend, text="■ OS/Apps", fg="#999999").pack(side="left", padx=10)
        tk.Label(legend, text="■ Model Total", fg="#3366cc").pack(side="left", padx=10)
        tk.Label(legend, text="■ Projected Free", fg="#00CC00").pack(side="left", padx=10)

        # ---------------- 6. Buttons ----------------
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=7, column=0, columnspan=3, pady=5)
        self.run_btn = tk.Button(btn_frame, text="Run Inference",
                                 command=lambda: self.start_task("run"),
                                 bg="#28a745", fg="white", width=12)
        self.run_btn.pack(side="left", padx=5)
        self.inspect_btn = tk.Button(btn_frame, text="Inspect Model",
                                     command=lambda: self.start_task("inspect"),
                                     bg="#007bff", fg="white", width=12)
        self.inspect_btn.pack(side="left", padx=5)
        tk.Button(btn_frame, text="Stop", command=self.stop_process,
                  bg="#dc3545", fg="white", width=8).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save Log", command=self.save_log, width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Copy Result Card", command=self.copy_result_card,
                  bg="#0a66c2", fg="white", width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Clear All", command=self.clear_all, width=10).pack(side="left", padx=5)

        self.status_label = tk.Label(root, text="System: Monitoring... | PID: None", fg="grey")
        self.status_label.grid(row=8, column=0, columnspan=3)

        self.output_text = tk.Text(root, height=10, width=120, font=("Consolas", 9))
        self.output_text.grid(row=9, column=0, columnspan=3, padx=10, pady=5)

        # ---------------- 7. Paid-suite teaser ----------------
        teaser = tk.Frame(root)
        teaser.grid(row=10, column=0, columnspan=3, sticky="ew", padx=10, pady=(2, 8))
        tk.Label(teaser, text="🔒 What can I run on this PC?",
                 font=("Arial", 9, "bold")).pack(side="left")
        tk.Button(teaser, text="Get model recommendations (CPU AI Lab suite)",
                  command=self.show_teaser, state="disabled",
                  bg="#cccccc").pack(side="left", padx=10)
        tk.Label(teaser, text="free preview coming soon", fg="grey",
                 font=("Arial", 8, "italic")).pack(side="left")

        self.current_model_mib = self.current_kv_mib = self.current_compute_mib = 0.0
        self.current_mode = "run"
        self.offload_txt = None
        self.vram_txt = None
        self.draw_ram_bar(0)

        threading.Thread(target=self.setup_engine, daemon=True).start()

    # ---------------------------------------------------------------
    # Engine provisioning + Vulkan detection
    # ---------------------------------------------------------------
    def setup_engine(self):
        try:
            if self.cli_path is None:
                self.set_engine_status("Engine: downloading pinned llama.cpp "
                                       f"{PINNED_BUILD} (~31 MB)...")
                self.provision_engine()
                self.cli_path = find_cli()
            if self.cli_path is None:
                self.set_engine_status("Engine: MISSING - engine download failed")
                return

            ver = engine_version(self.cli_path)
            self.set_engine_status(f"Engine: llama.cpp {PINNED_BUILD} | {ver}")

            dll_ok, device = detect_vulkan(self.cli_path)
            self.vulkan_ok = bool(dll_ok and device)
            self.vulkan_device = device
            if self.vulkan_ok:
                self.ui(lambda: self.apply_gpu_mode(True, device))
            else:
                reason = "no Vulkan loader (vulkan-1.dll)" if not dll_ok else \
                    "loader present, but engine found no Vulkan device"
                self.ui(lambda: self.apply_gpu_mode(False, reason))
        except Exception as e:
            self.set_engine_status(f"Engine: setup error - {e}")

    def set_engine_status(self, text):
        self.ui(lambda: self.engine_status.set(text))

    def ui(self, fn):
        """Schedule a callable on the Tk main thread (thread-safe)."""
        self.ui_queue.put(fn)

    def _pump_ui(self):
        try:
            while True:
                fn = self.ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.root.after(80, self._pump_ui)

    def apply_gpu_mode(self, on, info):
        if on:
            self.gpu_status.set(f"GPU: {info} -> offload ON (-ngl 99)")
            self.ngl.set(99)
            self.ngl_spin.config(state="normal")
        else:
            self.gpu_status.set(f"GPU: CPU-only mode ({info})")
            self.ngl.set(0)
            self.ngl_spin.config(state="disabled")

    def provision_engine(self):
        """Download + SHA256-verify + extract the pinned engine zip."""
        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = ENGINE_DIR / f"llama-{PINNED_BUILD}-bin-win-vulkan-x64.zip"
        if not zip_path.exists():
            req = urllib.request.Request(PINNED_URL, headers={"User-Agent": "CPU-AI-Lab-Bench"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(zip_path, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
        digest = sha256_file(zip_path)
        if digest != PINNED_SHA256:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"engine zip SHA256 mismatch (got {digest[:16]}...) - "
                               "download aborted for your safety")
        self.set_engine_status("Engine: verifying OK, extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ENGINE_DIR)
        zip_path.unlink(missing_ok=True)
        if not (ENGINE_DIR / "llama-cli.exe").exists():
            raise RuntimeError("llama-cli.exe missing after extraction")

    # ---------------------------------------------------------------
    # Test-model download (tiered by RAM, stacked like the installer)
    # ---------------------------------------------------------------
    def download_test_model(self):
        if self.total_sys_ram_gb >= 12:
            tier, models = "High (High + Mid + Low)", HIGH_TIER + MID_TIER + LOW_TIER
        elif self.total_sys_ram_gb >= 9:
            tier, models = "Medium (Mid + Low)", MID_TIER + LOW_TIER
        else:
            tier, models = "Low", LOW_TIER
        name, url = models[0]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / url.split("/")[-1].split("?")[0]
        if dest.exists():
            self.model_path.set(str(dest))
            messagebox.showinfo("Already have it", f"{name} is already downloaded.\nSelected.")
            return
        self.get_model_btn.config(state="disabled", text=f"Downloading {name}...")
        threading.Thread(target=self._fetch_model, args=(url, dest, name, tier),
                         daemon=True).start()

    def _fetch_model(self, url, dest, name, tier):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CPU-AI-Lab-Bench"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            def done():
                self.model_path.set(str(dest))
                self.get_model_btn.config(state="normal", text="Get test model")
                messagebox.showinfo(
                    "Ready to benchmark",
                    f"{name} downloaded (tier: {tier} for "
                    f"{self.total_sys_ram_gb:.0f} GB RAM).\n"
                    "Press INSPECT first, then RUN INference.")
            self.ui(done)
        except Exception as e:
            def fail():
                self.get_model_btn.config(state="normal", text="Get test model")
                messagebox.showerror("Download failed", str(e))
            self.ui(fail)

    # ---------------------------------------------------------------
    # UI helpers (ported)
    # ---------------------------------------------------------------
    def browse_file(self):
        fn = filedialog.askopenfilename(filetypes=[("GGUF files", "*.gguf")])
        if fn:
            self.model_path.set(fn)

    def browse_prompt_file(self):
        fn = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if fn:
            self.prompt_file_path.set(fn)
            self.custom_prompt.delete("1.0", tk.END)
            self.custom_prompt.insert("1.0", f"[File Mode Active: {os.path.basename(fn)}]")
            self.custom_prompt.config(state="disabled", bg="#f0f0f0")

    def parse_line(self, line):
        if self.current_mode == "run" and "Prompt:" in line and "Generation:" in line:
            m = re.search(r"Prompt:\s+([\d\.]+)\s+t/s\s+\|\s+Generation:\s+([\d\.]+)\s+t/s", line)
            if m:
                self.stat_prompt_speed.set(f"Prompt: {m.group(1)} t/s")
                self.stat_gen_speed.set(f"Gen: {m.group(2)} t/s")
                res = self.last_result or {}
                res.update({"prompt_tps": m.group(1), "gen_tps": m.group(2)})
                self.last_result = res
                self.update_verdict(float(m.group(2)))

        if "Host" in line and "=" in line and "+" in line:
            m = re.search(r"\|\s+-\s+Host\s+\|\s+\d+\s+=\s+([\d\.]+)\s+\+\s+([\d\.]+)\s+\+\s+([\d\.]+)", line)
            if m:
                self.current_model_mib = float(m.group(1))
                self.current_kv_mib = float(m.group(2))
                self.current_compute_mib = float(m.group(3))
                self.stat_model_ram.set(f"Weights: {self.current_model_mib:.2f} MiB")
                self.stat_kv_ram.set(f"KV Cache: {self.current_kv_mib:.2f} MiB")
                self.stat_buffer_ram.set(f"Compute Buffer: {self.current_compute_mib:.2f} MiB")

        if "n_layer" in line:
            m = re.search(r"n_layer\s+=\s+(\d+)", line)
            if m:
                self.stat_layers.set(f"Layers: {m.group(1)}")

        # b11669 trace-level lines (new vs v1.6.5)
        if "offloaded" in line and "layers to GPU" in line:
            m = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", line)
            if m:
                self.offload_txt = f"{m.group(1)}/{m.group(2)} layers"
                self.stat_offload.set(f"Offload: {m.group(1)}/{m.group(2)} layers -> GPU")

        if "Vulkan" in line and "(" in line and "=" in line:
            # | - Vulkan0 (Radeon 780M) | 16724 = 15073 + ( 601 = 497 + 67 + 37) + 1049 |
            m = re.search(r"\(\s*(\d+)\s+=\s+(\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)\s*\)", line)
            if m:
                v_tot, v_w, v_kv, v_c = (int(g) for g in m.groups())
                self.vram_txt = f"{v_tot} MiB ({v_w}+{v_kv}+{v_c})"
                self.stat_vram.set(f"VRAM: {v_tot} MiB ({v_w}+{v_kv}+{v_c})")

        if "model buffer size" in line:
            m = re.search(r"size\s+=\s+([\d\.]+)", line)
            if m and float(m.group(1)) > self.current_model_mib:
                self.current_model_mib = float(m.group(1))
                self.stat_model_ram.set(f"Weights: {self.current_model_mib:.2f} MiB")

        if "KV buffer size" in line:
            m = re.search(r"size\s+=\s+([\d\.]+)", line)
            if m and float(m.group(1)) > self.current_kv_mib:
                self.current_kv_mib = float(m.group(1))
                self.stat_kv_ram.set(f"KV Cache: {self.current_kv_mib:.2f} MiB")

        if "compute buffer size" in line:
            m = re.search(r"size\s+=\s+([\d\.]+)", line)
            if m and float(m.group(1)) > self.current_compute_mib:
                self.current_compute_mib = float(m.group(1))
                self.stat_buffer_ram.set(f"Compute Buffer: {self.current_compute_mib:.2f} MiB")

        self.update_total_ram_display()

    def update_verdict(self, gen_tps: float):
        if gen_tps >= 30:
            v = "SNAPPY - great for daily use"
        elif gen_tps >= 12:
            v = "USABLE - comfortable pace"
        elif gen_tps >= 5:
            v = "SLOW - needs a smaller model"
        else:
            v = "TOO SLOW - this model is too big for this PC"
        self.verdict_var.set(f"Verdict: {v}")
        if self.last_result is not None:
            self.last_result["verdict"] = v
        else:
            self.last_result = {"verdict": v}

    def update_total_ram_display(self):
        total_gb = (self.current_model_mib + self.current_kv_mib + self.current_compute_mib) / 1024
        self.stat_total_ram.set(f"Model Est. Total: {total_gb:.2f} GB")
        mem = psutil.virtual_memory()
        usage_ratio = (total_gb + (mem.total - mem.available) / (1024 ** 3)) / self.total_sys_ram_gb
        self.ram_display.config(fg="red" if usage_ratio > 0.9 else "orange" if usage_ratio > 0.7 else "black")
        self.draw_ram_bar(total_gb)

    def draw_ram_bar(self, est_model_gb):
        self.canvas.delete("all")
        mem = psutil.virtual_memory()
        used_now_gb = (mem.total - mem.available) / (1024 ** 3)
        px_used = (used_now_gb / self.total_sys_ram_gb) * self.canvas_width
        px_model = (est_model_gb / self.total_sys_ram_gb) * self.canvas_width
        self.canvas.create_rectangle(0, 0, self.canvas_width, 25, fill="#00CC00", outline="")
        self.canvas.create_rectangle(0, 0, px_used, 25, fill="#999999", outline="")
        color = "#3366cc"
        if (used_now_gb + est_model_gb) > self.total_sys_ram_gb:
            color = "#cc0000"
        self.canvas.create_rectangle(px_used, 0, min(self.canvas_width, px_used + px_model),
                                     25, fill=color, outline="")
        future_free = max(0, self.total_sys_ram_gb - (used_now_gb + est_model_gb))
        self.stat_free_ram.set(f"System Available: {mem.available / (1024 ** 3):.2f} GB "
                               f"(Projected: {future_free:.2f} GB)")

    # ---------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------
    def build_cmd(self, mode):
        """Assemble the llama-cli command for the PINNED b11669 engine.

        Changes vs v1.6.5:
          - '--mmap' REMOVED (flag gone in 0.7.0; load-mode auto already mmaps)
          - '-ngl' passed ONLY as 0 or 99 based on engine-confirmed Vulkan
          - prompt passed raw (no manual quote wrapping)
        """
        cmd = [str(self.cli_path),
               "-m", self.model_path.get(),
               "-t", str(self.threads.get()),
               "-c", str(self.context.get()),
               "-st", "-lv", "4"]  # trace level: memory breakdown + n_layer + offload lines

        # Vulkan gate: explicit value either way (deterministic, honest)
        cmd.extend(["-ngl", "99" if (self.vulkan_ok and self.ngl.get() > 0)
                    else "0"])

        cmd.extend(["--reasoning-budget", str(self.reasoning_budget.get())])

        if self.ctk_var.get():
            cmd.extend(["-ctk", "q8_0"])
        if self.ctv_var.get():
            cmd.extend(["-ctv", "q8_0"])
        if self.fa_var.get():
            cmd.extend(["-fa", "on"])

        if mode == "inspect":
            cmd.extend(["-n", "0", "-p", "Hi!", "--no-display-prompt"])
        else:
            if self.prompt_file_path.get():
                cmd.extend(["-f", self.prompt_file_path.get()])
            else:
                user_p = self.custom_prompt.get("1.0", tk.END).strip()
                cmd.extend(["-p", user_p])
            cmd.extend(["-n", str(self.n_predict.get())])
        return cmd

    def _handle_line(self, line):
        self.output_text.insert(tk.END, line)
        self.output_text.see(tk.END)
        self.parse_line(line)

    def execute_command(self, cmd, is_inference):
        try:
            self.current_model_mib = self.current_kv_mib = self.current_compute_mib = 0.0
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                creationflags=0x08000000, errors="replace")
            pid = self.process.pid
            self.ui(lambda: self.status_label.config(
                text=f"Total System RAM: {self.total_sys_ram_gb:.2f} GB | PID: {pid}"))

            while not self.is_stopping:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    self.ui(lambda l=line: self._handle_line(l))

            if self.process:
                self.process.stdout.close()
                self.process.wait()

            self.ui(lambda: self.status_label.config(
                text=f"Total System RAM: {self.total_sys_ram_gb:.2f} GB | PID: None"))
            self.is_stopping = False
            self.ui(lambda: (self.run_btn.config(state="normal"),
                             self.inspect_btn.config(state="normal")))
        except Exception as e:
            self.ui(lambda: messagebox.showerror("Error", str(e)))
            self.is_stopping = False
            self.ui(lambda: (self.run_btn.config(state="normal"),
                             self.inspect_btn.config(state="normal")))

    def stop_process(self):
        if self.process:
            self.is_stopping = True
            self.process.terminate()
            self.output_text.insert(tk.END, "\n[User Terminated Process]\n")

    def start_task(self, mode):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Busy", "A task is already running.")
            return
        if self.cli_path is None:
            return messagebox.showerror("Engine missing",
                                        "Engine not ready yet - wait for the status line, "
                                        "or re-run setup.")
        if not self.model_path.get():
            return messagebox.showerror("Error",
                                        "Select a model (or press 'Get test model').")

        self.current_mode = mode
        cmd = self.build_cmd(mode)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(
            tk.END,
            f"[CPU AI Lab] engine=llama.cpp {PINNED_BUILD} | "
            f"gpu={'ON (-ngl 99)' if (self.vulkan_ok and self.ngl.get() > 0) else 'OFF (-ngl 0)'} | "
            f"mode={mode}\n"
            "[CPU AI Lab] TIP: the FIRST GPU run includes shader warm-up - "
            "run it twice and trust the second result.\n" + "-" * 70 + "\n")
        self.run_btn.config(state="disabled")
        self.inspect_btn.config(state="disabled")
        threading.Thread(target=self.execute_command,
                         args=(cmd, mode == "run"), daemon=True).start()

    # ---------------------------------------------------------------
    # Result card / logs / extras
    # ---------------------------------------------------------------
    def copy_result_card(self):
        mem = psutil.virtual_memory()
        res = self.last_result or {}
        gpu = (f"Vulkan: {self.vulkan_device} (-ngl 99)" if self.vulkan_ok
               else "CPU only (-ngl 0)")
        model = os.path.basename(self.model_path.get()) if self.model_path.get() else "-"
        card = "\n".join([
            f"{APP_NAME} v{APP_VERSION}",
            f"Model: {model} | Context: {self.context.get()} | Threads: {self.threads.get()}",
            f"CPU: {psutil.cpu_count(logical=True)} threads | "
            f"RAM: {mem.total / (1024 ** 3):.0f} GB | GPU: {gpu}",
            f"Offload: {self.offload_txt or '-'} | VRAM: {self.vram_txt or '-'}",
            f"Engine: llama.cpp {PINNED_BUILD}",
            f"Prompt: {res.get('prompt_tps', '-')} t/s | "
            f"Generation: {res.get('gen_tps', '-')} t/s",
            f"Verdict: {res.get('verdict', '-')}",
            f"Tested: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ])
        self.root.clipboard_clear()
        self.root.clipboard_append(card)
        messagebox.showinfo("Result card copied",
                            "Result card copied to clipboard - paste it anywhere "
                            "(forum, Reddit, Discord):\n\n" + card)

    def show_teaser(self):
        messagebox.showinfo(
            "CPU AI Lab suite",
            "The recommender is coming with the full CPU AI Lab suite:\n\n"
            "it reads your benchmark result and tells you exactly WHICH models "
            "this PC can run, and how fast.\n\n"
            "Your free bench result is the first step.")

    def save_log(self):
        content = self.output_text.get("1.0", tk.END).strip()
        if not content:
            return messagebox.showwarning("Empty", "No log content to save.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CPU_AI_Lab_Bench_{timestamp}.txt"
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 initialfile=filename)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Saved", f"Log saved to:\n{file_path}")

    def clear_all(self):
        self.output_text.delete(1.0, tk.END)
        self.prompt_file_path.set("")
        self.custom_prompt.config(state="normal", bg="white")
        self.custom_prompt.delete("1.0", tk.END)
        self.custom_prompt.insert(
            "1.0", "Explain the difference between CPU and GPU inference in simple terms.")
        self.stat_prompt_speed.set("Prompt: -")
        self.stat_gen_speed.set("Gen: -")
        self.verdict_var.set("Verdict: -")
        self.last_result = None
        self.offload_txt = None
        self.vram_txt = None
        self.stat_offload.set("Offload: -")
        self.stat_vram.set("VRAM: -")
        self.current_model_mib = self.current_kv_mib = self.current_compute_mib = 0.0
        self.update_total_ram_display()


if __name__ == "__main__":
    root = tk.Tk()
    app = CpuLabBench(root)
    root.mainloop()
