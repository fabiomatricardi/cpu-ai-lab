#!/usr/bin/env python3
"""
CPU AI Lab - Launcher  (Phase 2: paid suite core UI)

Bench tab reuse + server tab: pick a curated preset from src/models.json,
compose the llama-server command PROGRAMMATICALLY (D27 - no duplicate flags),
and run it with the security defaults baked in (D13/D14):

    --host 127.0.0.1 (localhost only)  ·  --tools NEVER emitted  ·
    --ui + --ui-mcp-proxy (web chat + Exa MCP proxy, localhost only)

Modes:
    Chat     - any preset runs (engine does its structured-output best)
    Harness  - D31: target Q6_K + KV -ctk/-ctv q4_0, presets at/below Q4_K_M
               are greyed with the reason shown (chat-only presets stay
               available in Chat mode)

Reasoning (D30): ONE GUI control (Off/Low/Medium/High/XHigh) <-> token budget;
the launcher emits ONLY --reasoning-budget N.  --reasoning-effort is NEVER
emitted.  Default port 11434 with port-busy detection (D34).

Gate:  python tools/verify_flags.py --server   (must PASS before building)
Smoke: python tests/launcher_smoke.py
"""

import json
import os
import queue
import shlex
import socket
import subprocess
import sys
import threading
import tkinter as tk
import urllib.request
import webbrowser
from collections import OrderedDict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_NAME = "CPU AI Lab - Launcher"
APP_VERSION = "0.2.0-dev"

# --- Pinned engine (same contract as the bench) ----------------------------
PINNED_BUILD = "b11669"
PINNED_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b11669/llama-b11669-bin-win-vulkan-x64.zip"
)
PINNED_SHA256 = "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e"

# --- D30: the one reasoning control (level <-> token budget) ---------------
REASONING_LEVELS = OrderedDict([
    ("off",    ("Off",    0)),
    ("low",    ("Low",    1024)),
    ("medium", ("Medium", 2048)),
    ("high",   ("High",   4096)),
    ("xhigh",  ("XHigh",  8192)),
])
REASONING_SLACK = 4096            # OpenCode max_new_tokens = budget + slack (D26/D30)

# Chrome flags the launcher owns (gate: verify_flags.py --server manifest).
# Preset flags are merged in programmatically; duplicates collapse (D27).
DEFAULT_HOST = "127.0.0.1"        # D13: localhost only, always
DEFAULT_PORT = 11434              # D34
DEFAULT_NGL = 99
DEFAULT_FA = "on"
DEFAULT_LV = "4"
DEFAULT_NP = "1"


def app_dir() -> Path:
    """Folder for user data (engine/, models/): next to the .exe when frozen,
    otherwise the source folder. NOT the PyInstaller temp dir."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ENGINE_DIR = app_dir() / "engine"
MODELS_DIR = app_dir() / "models"


def find_server() -> Path | None:
    """Locate llama-server.exe (flat install layout or dev engine/vulkan layout)."""
    for cand in (ENGINE_DIR / "llama-server.exe",
                 app_dir() / "engine" / "vulkan" / "llama-server.exe",
                 Path(__file__).resolve().parent.parent / "engine" / "vulkan" / "llama-server.exe"):
        if cand.exists():
            return cand
    return None


def find_models_json() -> Path | None:
    """Locate the preset list (single source: src/models.json)."""
    here = Path(__file__).resolve().parent
    for cand in (here / "models.json",
                 app_dir() / "src" / "models.json",
                 app_dir() / "models.json",
                 app_dir() / "data" / "models.json"):
        if cand.exists():
            return cand
    return None


def load_presets() -> tuple[dict, str | None]:
    """Read models.json -> (payload, error). Empty payload on failure."""
    path = find_models_json()
    if path is None:
        return {}, "models.json not found"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("presets"), list) or not data["presets"]:
            return {}, "models.json has no presets"
        return data, None
    except Exception as exc:                    # malformed JSON etc.
        return {}, f"models.json unreadable: {exc}"


# ---------------------------------------------------------------------------
# D31: harness eligibility (target Q6_K, block Q4_K_M and below)
# ---------------------------------------------------------------------------

def quant_rank(quant: str) -> int:
    """Coarse weight-tier rank of a quant name: Q2_* -> 2 ... Q8_0 -> 8.
    Unknown/empty -> 0 (treated as blocked: never guess upward)."""
    q = (quant or "").upper().replace("Q", "", 1)
    for prefix in ("8", "6", "5", "4", "3", "2"):
        if q.startswith(prefix):
            return int(prefix)
    return 0


def harness_status(preset: dict, defaults: dict) -> dict:
    """eligible? + human reason. Uses defaults.harness policy (D31) verbatim -
    preferred Q6_K, blocked at/below Q4_K_M, KV remedy q4_0."""
    harness = defaults.get("harness", {})
    blocked_below = str(harness.get("blocked_at_or_below", "Q4_K_M"))
    target = str(harness.get("preferred_quant", "Q6_K"))
    kv = str(harness.get("kv_quant", "q4_0"))
    floor = quant_rank(blocked_below)           # Q4_K_M -> 4

    if preset.get("agent_ok"):                  # D33 escape hatch (reason required)
        return {"eligible": True, "escape": True, "rank": quant_rank(preset.get("quant", "")),
                "target": target, "kv": kv,
                "reason": str(preset.get("agent_reason") or "escape hatch (agent_ok)")}
    rank = quant_rank(preset.get("quant", ""))
    if rank >= 5:
        on_target = preset.get("quant", "").upper() == target.upper()
        hint = "" if on_target else " (usable, target is Q6_K)"
        return {"eligible": True, "escape": False, "rank": rank, "target": target, "kv": kv,
                "reason": ("on target " + target) if on_target else ("eligible" + hint)}
    return {"eligible": False, "escape": False, "rank": rank, "target": target, "kv": kv,
            "reason": (f"{preset.get('quant', '?')} is at/below {blocked_below}: structured output "
                       f"breaks down -> chat-only. Harness target {target} + KV -ctk/-ctv {kv} (D31)")}


# ---------------------------------------------------------------------------
# D27: programmatic command build (single merge point, no duplicate flags)
# ---------------------------------------------------------------------------

def _flag_pairs(seq) -> list[tuple[str, str | None]]:
    """['-c','64000','--jinja','--top-k','-1']
    -> [('-c','64000'), ('--jinja',None), ('--top-k','-1')]
    Negative numbers are values, not flags (-1 must stay with its flag)."""
    pairs, i = [], 0
    while i < len(seq):
        key = str(seq[i])
        if i + 1 < len(seq):
            nxt = str(seq[i + 1])
            looks_like_flag = nxt.startswith("--") or (
                nxt.startswith("-") and len(nxt) > 1 and not nxt[1:2].isdigit())
            if not looks_like_flag:
                pairs.append((key, nxt))
                i += 2
                continue
        pairs.append((key, None))
        i += 1
    return pairs


def compose_command(preset: dict, *, server: str, model_path: str,
                    port: int = DEFAULT_PORT, threads: int = 4, ngl: int = DEFAULT_NGL,
                    ctx: int | None = None, kv: tuple[str, str] | None = None,
                    reasoning: str = "medium", mcp_proxy: bool = True,
                    web_ui: bool = True) -> list[str]:
    """Build the llama-server argv. Chrome (launcher-owned) + preset flags merged
    into one ordered map: preset wins where it owns the flag (-c, KV, sampling),
    chrome supplies the security/port/UX flags. Duplicates impossible by construction."""
    merged: OrderedDict[str, str | None] = OrderedDict()

    # 1) preset flags first (they own context, KV defaults, sampling, --spec-*)
    for key, val in _flag_pairs(preset.get("flags", [])):
        merged[key] = val

    # 2) chrome defaults - setdefault: never overwrite what the preset owns
    merged.setdefault("-c", "32768")
    merged.setdefault("-ctk", "q4_0")
    merged.setdefault("-ctv", "q4_0")
    chrome = [
        ("-np", str(DEFAULT_NP)),
        ("-t", str(int(threads))),
        ("-ngl", str(int(ngl))),
        ("-fa", DEFAULT_FA),
        ("-lv", DEFAULT_LV),
        ("--jinja", None),
        ("--host", DEFAULT_HOST),           # D13: always localhost, not configurable
        ("--port", str(int(port))),
        ("--ui", None) if web_ui else None,
        ("--ui-mcp-proxy", None) if mcp_proxy else None,
        ("--alias", str(preset.get("alias") or preset.get("id", "model"))),
        ("--reasoning-budget", str(_budget_of(reasoning))),   # D30: ONLY reasoning flag
    ]
    for item in chrome:
        if item is None:
            continue
        key, val = item
        merged.setdefault(key, val)

    # 3) explicit UI overrides (ctx / KV controls) - replace value, keep position
    if ctx is not None:
        merged["-c"] = str(int(ctx))
    if kv is not None:
        merged["-ctk"], merged["-ctv"] = kv

    argv = [str(server), "-m", str(model_path)]
    for key, val in merged.items():
        argv.append(key)
        if val is not None:
            argv.append(val)
    return argv


def _budget_of(level: str) -> int:
    return REASONING_LEVELS.get(level, REASONING_LEVELS["medium"])[1]


def budget_to_level(budget: int | None) -> str:
    """Reverse map (OpenCode config or preset value -> GUI level name)."""
    if budget is None:
        return "medium"
    best, best_diff = "off", None
    for name, (_, value) in REASONING_LEVELS.items():
        diff = abs(int(budget) - value)
        if best_diff is None or diff < best_diff:
            best, best_diff = name, diff
    return best


def cmd_string(argv: list[str]) -> str:
    return shlex.join([str(a) for a in argv])


# ---------------------------------------------------------------------------
# D34: port-busy detection  ·  OpenCode export rules (D26/D30)
# ---------------------------------------------------------------------------

def port_busy(port: int, host: str = DEFAULT_HOST, timeout: float = 0.35) -> bool:
    """True when something already listens on host:port (e.g. Ollama on 11434)."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def opencode_export(preset: dict, *, argv: list[str], reasoning: str,
                    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """OpenCode pairing snippet (D26/D30):
    context_window = preset -c;  max_new_tokens = budget + slack (never <= budget)."""
    pairs = dict(_flag_pairs(argv[argv.index("-m") + 2:]))  # skip server, -m and the path
    ctx = int(pairs.get("-c") or 32768)
    budget = _budget_of(reasoning)
    temperature = pairs.get("--temperature") or pairs.get("--temp") or "1.0"
    top_p = pairs.get("--top-p", "0.95")
    cfg = {
        "provider": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "cpu-ai-lab",
            "options": {"baseURL": f"http://{host}:{port}/v1", "apiKey": "local"},
        },
        "model": str(preset.get("alias") or preset.get("id", "model")),
        "context_window": ctx,
        "max_new_tokens": budget + REASONING_SLACK,   # always > budget (D26)
        "reasoning": {"level": reasoning, "budget": budget, "flag": f"--reasoning-budget {budget}"},
        "sampling": {"temperature": temperature, "top_p": top_p},
        "command": cmd_string(argv),
        "note": "start the server first (launcher or the command above); "
                "OpenCode talks to 127.0.0.1 only, --tools stays off server-side",
    }
    return json.dumps(cfg, indent=2)


def download_file(url: str, dest: Path, progress=None) -> Path:
    """Direct GGUF download (D24). Streams to dest.with_suffix('.part') then renames."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cpu-ai-lab-launcher/0.2"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    part.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class CpuLabLauncher:
    POLL_MS = 80

    def __init__(self, root: tk.Tk):
        self.root = root
        self.ui_queue: "queue.Queue" = queue.Queue()
        self.server_proc: subprocess.Popen | None = None
        self.log_history: list[str] = []
        self.data, self.load_error = load_presets()
        self.presets: list[dict] = list(self.data.get("presets", []))
        self.defaults: dict = self.data.get("defaults", {})
        self.server_bin = find_server()
        self.vulkan = self._detect_vulkan()
        self.threads_default = max(1, min(4, (os.cpu_count() or 4) // 2))

        root.title(f"{APP_NAME}  ·  v{APP_VERSION}  ·  llama.cpp {PINNED_BUILD}")
        root.geometry("1080x760")
        root.minsize(940, 660)
        self._build_ui()
        root.after(self.POLL_MS, self._pump_ui)

        if self.load_error:
            messagebox.showerror(APP_NAME, self.load_error)
        elif self.presets:
            self._rebuild_tree()               # populate BEFORE selecting (iid must exist)
            self._select_default_preset()
        self.refresh()

    # ---- infrastructure ---------------------------------------------------
    def ui(self, fn):
        self.ui_queue.put(fn)

    def _pump_ui(self):
        try:
            while True:
                fn = self.ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.root.after(self.POLL_MS, self._pump_ui)

    def log(self, text: str):
        self.log_history.append(text)
        self.log_history = self.log_history[-400:]
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _detect_vulkan() -> bool:
        dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "vulkan-1.dll"
        return dll.exists()

    # ---- UI construction --------------------------------------------------
    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("TButton", padding=4)
        style.configure("Big.TButton", padding=6)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # header: title + engine badge + security banner (D13)
        head = ttk.Frame(outer)
        head.pack(fill="x")
        ttk.Label(head, text="CPU AI Lab Launcher", font=("Segoe UI", 15, "bold")).pack(side="left")
        gpu = "Vulkan ON" if self.vulkan else "Vulkan not detected"
        ttk.Label(head, text=f"llama.cpp {PINNED_BUILD} · {gpu}",
                  foreground="#555").pack(side="left", padx=12)
        ttk.Button(head, text="▶ Honest Bench (free tool)",
                   command=self.launch_bench).pack(side="right")
        banner = ("SECURITY: binds 127.0.0.1 only · --tools never enabled · "
                  "web-chat queries go to Exa (privacy banner shown in chat)")
        ttk.Label(outer, text=banner, foreground="#7a4b00",
                  background="#fff3d6", padding=(8, 4),
                  relief="groove").pack(fill="x", pady=(6, 8))

        # main panes: left = presets + mode, right = config + command + log
        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        # --- left: mode + preset list ------------------------------------
        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        body.add(left, weight=1)

        mode_box = ttk.LabelFrame(left, text="Mode", padding=8)
        mode_box.pack(fill="x")
        self.mode_var = tk.StringVar(value="chat")
        ttk.Radiobutton(mode_box, text="Chat (browser UI, any preset)",
                        variable=self.mode_var, value="chat",
                        command=self._on_mode_change).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="Harness / OpenCode (D31: Q6_K target)",
                        variable=self.mode_var, value="harness",
                        command=self._on_mode_change).pack(anchor="w")

        reason_box = ttk.LabelFrame(left, text="Reasoning control  (emits --reasoning-budget only, D30)",
                                    padding=8)
        reason_box.pack(fill="x", pady=(8, 0))
        row = ttk.Frame(reason_box)
        row.pack(fill="x")
        ttk.Label(row, text="Level:").pack(side="left")
        self.reasoning_var = tk.StringVar(value="medium")
        self.reasoning_cb = ttk.Combobox(row, state="readonly", width=9,
                                         textvariable=self.reasoning_var,
                                         values=list(REASONING_LEVELS))
        self.reasoning_cb.pack(side="left", padx=8)
        self.reasoning_var.trace_add("write", lambda *_: self.refresh())
        self.budget_lbl = ttk.Label(row, text="", font=("Consolas", 9))
        self.budget_lbl.pack(side="left")

        ttk.Label(reason_box, wraplength=300, foreground="#555",
                  text="Off=0 · Low=1024 · Medium=2048 · High=4096 · XHigh=8192 tokens. "
                       "Same knob maps to OpenCode max_new_tokens = budget + 4096."
                  ).pack(fill="x", pady=(6, 0))

        list_box = ttk.LabelFrame(left, text="Curated presets (src/models.json)", padding=4)
        list_box.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("quant", "ctx", "state")
        self.tree = ttk.Treeview(list_box, columns=cols, show="tree headings", height=10,
                                 selectmode="browse")
        self.tree.heading("#0", text="Preset")
        self.tree.heading("quant", text="Quant")
        self.tree.heading("ctx", text="Ctx")
        self.tree.heading("state", text="Harness")
        self.tree.column("#0", width=190, anchor="w")
        self.tree.column("quant", width=66, anchor="center")
        self.tree.column("ctx", width=64, anchor="center")
        self.tree.column("state", width=74, anchor="center")
        self.tree.tag_configure("blocked", foreground="#999")
        self.tree.tag_configure("target", foreground="#137333")
        scroll = ttk.Scrollbar(list_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.refresh())

        self.harness_note = ttk.Label(left, wraplength=320, foreground="#7a4b00")
        self.harness_note.pack(fill="x", pady=(6, 0))

        # --- right: config, command, actions, log ------------------------
        right = ttk.Frame(body)
        body.add(right, weight=2)

        cfg = ttk.LabelFrame(right, text="Model & server", padding=8)
        cfg.pack(fill="x")

        ttk.Label(cfg, text="Model file:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar()
        ttk.Entry(cfg, textvariable=self.model_var, width=64).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(cfg, text="Browse…", command=self.browse_model).grid(row=0, column=2)
        self.dl_btn = ttk.Button(cfg, text="Download", command=self.download_model)
        self.dl_btn.grid(row=0, column=3, padx=(4, 0))

        ttk.Label(cfg, text="Port:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.port_var = tk.StringVar(value=str(self.defaults.get("port", DEFAULT_PORT)))
        ttk.Entry(cfg, textvariable=self.port_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(cfg, text="Threads:").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.threads_var = tk.StringVar(value=str(self.threads_default))
        ttk.Entry(cfg, textvariable=self.threads_var, width=6).grid(row=1, column=3, sticky="w", padx=4, pady=(6, 0))

        ttk.Label(cfg, text="Context:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ctx_var = tk.StringVar(value="")
        ttk.Entry(cfg, textvariable=self.ctx_var, width=10).grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(cfg, foreground="#555",
                  text="blank = preset default · model path is yours to override (any GGUF on disk)"
                  ).grid(row=2, column=2, columnspan=2, sticky="w", pady=(6, 0))
        cfg.columnconfigure(1, weight=1)
        self.port_var.trace_add("write", lambda *_: self.refresh())
        self.threads_var.trace_add("write", lambda *_: self.refresh())
        self.ctx_var.trace_add("write", lambda *_: self.refresh())
        self.model_var.trace_add("write", lambda *_: self.refresh())

        # command preview (read-only) + actions
        cmd_box = ttk.LabelFrame(right, text="Command preview (programmatic build, D27)", padding=6)
        cmd_box.pack(fill="x", pady=(8, 0))
        self.cmd_text = tk.Text(cmd_box, height=5, wrap="word", font=("Consolas", 9),
                                state="disabled", background="#f6f8fa")
        self.cmd_text.pack(fill="x")

        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=8)
        self.start_btn = ttk.Button(actions, text="▶ Start server", style="Big.TButton",
                                    command=self.start_server)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="■ Stop", command=self.stop_server, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Copy command", command=self.copy_command).pack(side="left", padx=(12, 0))
        ttk.Button(actions, text="Open web chat", command=self.open_webchat).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="OpenCode config", command=self.copy_opencode).pack(side="left", padx=(6, 0))

        self.status_lbl = ttk.Label(right, text="idle", foreground="#555")
        self.status_lbl.pack(fill="x")

        log_box = ttk.LabelFrame(right, text="Server log", padding=4)
        log_box.pack(fill="both", expand=True, pady=(6, 0))
        self.log_text = tk.Text(log_box, height=12, state="disabled",
                                font=("Consolas", 9), background="#1e1e1e", foreground="#d4d4d4")
        self.log_text.pack(fill="both", expand=True)

        # footer
        ttk.Label(outer, foreground="#777",
                  text=f"Engine pinned {PINNED_BUILD} · SHA256 verified on install · "
                       f"presets gated by tools/verify_flags.py --server · {APP_VERSION}"
                  ).pack(fill="x", pady=(6, 0))

    # ---- state helpers ----------------------------------------------------
    def _selected_preset(self) -> dict | None:
        sel = self.tree.selection()
        if not sel:
            return None
        pid = sel[0]
        for p in self.presets:
            if p.get("id") == pid:
                return p
        return None

    def _threads(self) -> int:
        try:
            return max(1, int(self.threads_var.get()))
        except ValueError:
            return self.threads_default

    def _select_default_preset(self):
        mode = self.mode_var.get()
        for p in self.presets:
            if not self.tree.exists(p["id"]):
                continue
            if mode != "harness" or harness_status(p, self.defaults).get("eligible"):
                self.tree.selection_set(p["id"])
                self.tree.see(p["id"])
                return
        if self.presets and self.tree.exists(self.presets[0]["id"]):
            self.tree.selection_set(self.presets[0]["id"])

    def _rebuild_tree(self):
        mode = self.mode_var.get()
        self.tree.delete(*self.tree.get_children())
        first_eligible = None
        for p in self.presets:
            st = harness_status(p, self.defaults)
            eligible = st["eligible"] or mode != "harness"
            if mode == "harness" and st["eligible"] and first_eligible is None:
                first_eligible = p["id"]
            tags = ()
            if mode == "harness" and not st["eligible"]:
                tags = ("blocked",)
            elif p.get("quant", "").upper() == st["target"].upper():
                tags = ("target",)
            ctx = next((v for k, v in _flag_pairs(p.get("flags", [])) if k == "-c"), "?")
            self.tree.insert("", "end", iid=p["id"], text=p.get("id", "?"),
                             values=(p.get("quant", "?"), ctx,
                                     "ON" if st["eligible"] else "blocked"),
                             tags=tags)
        # keep selection valid across mode flips
        cur = self.tree.selection()
        if not cur or not self.tree.exists(cur[0]) or (
                mode == "harness" and not self._eligible_current()):
            self.tree.selection_set(first_eligible or self.presets[0]["id"])

    def _eligible_current(self) -> bool:
        p = self._selected_preset()
        return bool(p) and harness_status(p, self.defaults)["eligible"]

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self._rebuild_tree()
        p = self._selected_preset()
        if p is not None:
            st = harness_status(p, self.defaults)
            if mode == "harness" and not st["eligible"]:
                self.harness_note.configure(text="BLOCKED in Harness: " + st["reason"])
            elif st["escape"]:
                self.harness_note.configure(text="Escape hatch: " + st["reason"])
            elif mode == "harness":
                self.harness_note.configure(text=f"Harness OK: {st['reason']} · KV -ctk/-ctv {st['kv']}")
            else:
                self.harness_note.configure(text="")
        else:
            self.harness_note.configure(text="")
        self.refresh()

    # ---- the one place the command is built --------------------------------
    def _current_argv(self) -> list[str] | None:
        preset = self._selected_preset()
        if preset is None:
            return None
        try:
            port = int(self.port_var.get())
        except ValueError:
            port = DEFAULT_PORT
        threads = self._threads()
        ctx = None
        if self.ctx_var.get().strip().isdigit():
            ctx = int(self.ctx_var.get().strip())
        kv = None
        if self.mode_var.get() == "harness":            # D31 remedy always on in Harness
            remedy = harness_status(preset, self.defaults)["kv"]
            kv = (remedy, remedy)
        return compose_command(
            preset,
            server=str(self.server_bin or "llama-server.exe"),
            model_path=self.model_var.get().strip() or "(select a GGUF)",
            port=port, threads=threads,
            ngl=DEFAULT_NGL if self.vulkan else 0,
            ctx=ctx, kv=kv,
            reasoning=self.reasoning_var.get(),
        )

    def refresh(self):
        budget = _budget_of(self.reasoning_var.get())
        self.budget_lbl.configure(text=f"→ --reasoning-budget {budget}")
        argv = self._current_argv()
        if argv is not None:
            self._set_cmd(cmd_string(argv))
        preset = self._selected_preset()
        if preset is not None:
            # download button (D24) - enabled only when a URL exists (never guessed)
            url = preset.get("download_url")
            self.dl_btn.configure(state="normal" if url else "disabled")
            st = harness_status(preset, self.defaults)
            if self.mode_var.get() == "harness" and not st["eligible"]:
                self.start_btn.configure(state="disabled")
                if not self.harness_note.cget("text").startswith("BLOCKED"):
                    self.harness_note.configure(text="BLOCKED in Harness: " + st["reason"])
            else:
                self.start_btn.configure(state="disabled" if self.server_proc else "normal")
            if self.mode_var.get() == "harness" and st["eligible"]:
                self.harness_note.configure(
                    text=f"Harness OK: {st['reason']} · KV -ctk/-ctv {st['kv']}"
                )
            elif self.mode_var.get() == "chat":
                self.harness_note.configure(text="")
            # model path suggestion: preset filename if the entry is untouched
            if not self.model_var.get():
                guess = MODELS_DIR / str(preset.get("name") or "")
                if guess.exists():
                    self.model_var.set(str(guess))
        else:
            self.dl_btn.configure(state="disabled")
            self.start_btn.configure(state="disabled")

    def _set_cmd(self, text: str):
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", text)
        self.cmd_text.configure(state="disabled")

    # ---- actions -----------------------------------------------------------
    def launch_bench(self):
        """Bench reuse (2.2): start the free Honest Bench app alongside."""
        for cand in (app_dir() / "CPU_AI_Lab_Bench.exe",
                     app_dir() / "dist" / "CPU_AI_Lab_Bench.exe"):
            if cand.exists():
                subprocess.Popen([str(cand)],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.status_lbl.configure(text=f"launched {cand.name}")
                return
        src = app_dir() / "src" / "cpulab_bench.py"
        if src.exists():
            subprocess.Popen([sys.executable, str(src)])
            self.status_lbl.configure(text="launched bench from source")
            return
        messagebox.showinfo(APP_NAME,
                            "Honest Bench not found.\nExpected CPU_AI_Lab_Bench.exe "
                            "next to the launcher (or src/cpulab_bench.py).")

    def browse_model(self):
        initial = MODELS_DIR if MODELS_DIR.exists() else app_dir()
        path = filedialog.askopenfilename(
            title="Select a GGUF model", initialdir=str(initial),
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def download_model(self):
        preset = self._selected_preset()
        if preset is None:
            return
        url = preset.get("download_url")
        if not url:
            messagebox.showinfo(APP_NAME, "No download URL for this preset yet (D24 pending).")
            return
        dest = MODELS_DIR / str(preset.get("name") or "model.gguf")
        if dest.exists():
            if not messagebox.askyesno(APP_NAME, f"{dest.name} already exists. Re-download?"):
                return
        self.dl_btn.configure(state="disabled")
        self.status_lbl.configure(text=f"downloading {dest.name}…")

        def prog(done, total):
            pct = f"{100 * done // total}%" if total else "…"
            self.ui(lambda: self.status_lbl.configure(
                text=f"downloading {dest.name}: {done // (1 << 20)} MB {pct}"))

        def work():
            try:
                download_file(url, dest, prog)
                self.ui(lambda: self.status_lbl.configure(text=f"downloaded {dest.name}"))
                self.ui(lambda: self.model_var.set(str(dest)))
            except Exception as exc:
                self.ui(lambda: messagebox.showerror(APP_NAME, f"Download failed:\n{exc}"))
                self.ui(lambda: self.status_lbl.configure(text="download failed"))
            self.ui(lambda: self.dl_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def copy_command(self):
        argv = self._current_argv()
        if argv:
            self.root.clipboard_clear()
            self.root.clipboard_append(cmd_string(argv))
            self.status_lbl.configure(text="command copied to clipboard")

    def copy_opencode(self):
        preset, argv = self._selected_preset(), self._current_argv()
        if preset is None or argv is None:
            return
        try:
            port = int(self.port_var.get())
        except ValueError:
            port = DEFAULT_PORT
        text = opencode_export(preset, argv=argv, reasoning=self.reasoning_var.get(), port=port)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_lbl.configure(text="OpenCode config copied (max_new_tokens = budget + 4096)")

    def open_webchat(self):
        """Poll the port; when the server answers, open the built-in web chat."""
        try:
            port = int(self.port_var.get())
        except ValueError:
            port = DEFAULT_PORT
        self.status_lbl.configure(text=f"waiting for http://127.0.0.1:{port} …")

        def wait_open(deadline_s: float = 60.0):
            import time as _t
            t0 = _t.time()
            while _t.time() - t0 < deadline_s:
                if not port_busy(port):
                    _t.sleep(0.4)                      # brief grace after connect
                else:
                    _t.sleep(0.5)
                if port_busy(port):
                    self.ui(lambda: webbrowser.open(f"http://127.0.0.1:{port}/"))
                    self.ui(lambda: self.status_lbl.configure(text="web chat opened"))
                    return
                _t.sleep(0.5)
            self.ui(lambda: self.status_lbl.configure(
                text=f"nothing answered on port {port} - start the server first"))

        if port_busy(port):
            webbrowser.open(f"http://127.0.0.1:{port}/")
            self.status_lbl.configure(text="web chat opened")
        else:
            threading.Thread(target=wait_open, daemon=True).start()

    def start_server(self):
        if self.server_proc:
            return
        if self.server_bin is None:
            messagebox.showerror(
                APP_NAME, "llama-server.exe not found.\n"
                          "Expected in engine/ (flat install) or engine/vulkan/.")
            return
        preset = self._selected_preset()
        if preset is None:
            messagebox.showinfo(APP_NAME, "Select a preset first.")
            return
        if self.mode_var.get() == "harness" and not harness_status(preset, self.defaults)["eligible"]:
            messagebox.showwarning(APP_NAME,
                                   "Harness blocked (D31):\n"
                                   + harness_status(preset, self.defaults)["reason"])
            return
        model = self.model_var.get().strip()
        if not model or not Path(model).exists():
            messagebox.showinfo(APP_NAME, "Select a model file first (Browse… or Download).")
            return
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showwarning(APP_NAME, "Port must be a number.")
            return
        # D34: port-busy detection BEFORE launch (Ollama also claims 11434)
        if port_busy(port):
            busy_port = port
            if messagebox.askyesno(
                    APP_NAME,
                    f"Port {busy_port} is already in use (Ollama also defaults to 11434).\n"
                    f"Pick the next free port ({busy_port + 1}) and start there?"):
                port = busy_port + 1
                self.port_var.set(str(port))
            else:
                return

        argv = compose_command(
            preset, server=str(self.server_bin), model_path=model,
            port=port, threads=self._threads(),
            ngl=DEFAULT_NGL if self.vulkan else 0,
            ctx=int(self.ctx_var.get()) if self.ctx_var.get().strip().isdigit() else None,
            kv=((harness_status(preset, self.defaults)["kv"],) * 2
                if self.mode_var.get() == "harness" else None),
            reasoning=self.reasoning_var.get(),
        )
        self._set_cmd(cmd_string(argv))
        self.log("$ " + cmd_string(argv))
        try:
            self.server_proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.server_proc = None
            messagebox.showerror(APP_NAME, f"Could not start server:\n{exc}")
            return
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_lbl.configure(
            text=f"running on http://127.0.0.1:{port} · alias={preset.get('alias')} · --tools OFF")
        threading.Thread(target=self._read_log, daemon=True).start()
        threading.Thread(target=self._wait_exit, daemon=True).start()

    def _read_log(self):
        proc = self.server_proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.ui(lambda ln=line.rstrip(): self.log(ln))

    def _wait_exit(self):
        proc = self.server_proc
        if proc is None:
            return
        code = proc.wait()
        def done():
            self.log(f"[server exited with code {code}]")
            self.server_proc = None
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_lbl.configure(text=f"server stopped (code {code})")
        self.ui(done)

    def stop_server(self):
        proc = self.server_proc
        if proc is None:
            return
        self.log("[stop requested]")
        try:
            proc.terminate()
        except Exception:
            pass


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    CpuLabLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
