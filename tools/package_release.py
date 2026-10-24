#!/usr/bin/env python3
"""CPU AI Lab - release packager (item 2.6).

Assembles the customer deliverable zip from the manifest locked 2026-10-23
(D39): both exes, editable models.json, the UNMODIFIED pinned engine,
guides, terms, integrity manifests. No models bundled (D39: download on
first run). Build-side tools/tests/specs never ship.

    python tools/package_release.py                  # stage + zip + frozen selftest
    python tools/package_release.py --verify-engine PATH.zip
        # additionally prove dist/engine == that official release zip,
        # byte for byte (run once per re-pin; result recorded in VERSION.json)

Acceptance: the STAGED launcher passes --selftest (engine resolves next to
the exe, presets load from the shipped models.json).
Exit 0 = package is good.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
STAGE_ROOT = ROOT / "build" / "package"
SRC = ROOT / "src"
PACK = ROOT / "packaging"

EXES = ["CPU_AI_Lab_Launcher.exe", "CPU_AI_Lab_Bench.exe"]
PINNED_ZIP_NAME = "llama-b11669-bin-win-vulkan-x64.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_kv(pyfile: Path, key: str) -> str | None:
    m = re.search(rf'^{key}\s*=\s*"([^"]+)"', pyfile.read_text(encoding="utf-8"),
                  re.M)
    return m.group(1) if m else None


def verify_engine_zip(engine_dir: Path, zip_path: Path, pinned_sha: str) -> list[str]:
    """dist/engine must be the byte-identical content of the official zip."""
    problems = []
    actual = sha256_file(zip_path)
    if actual != pinned_sha:
        return [f"engine zip sha mismatch: {actual} != pinned {pinned_sha}"]
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if not n.endswith("/"))
        on_disk = sorted(p.name for p in engine_dir.iterdir() if p.is_file())
        if names != on_disk:
            missing = sorted(set(names) - set(on_disk))
            extra = sorted(set(on_disk) - set(names))
            if missing:
                problems.append(f"missing from dist/engine: {missing}")
            if extra:
                problems.append(f"extra in dist/engine: {extra}")
        for n in names:
            p = engine_dir / n
            if p.exists() and sha256_bytes(z.read(n)) != sha256_file(p):
                problems.append(f"content differs: {n}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-engine", metavar="ZIP",
                    help=f"prove dist/engine == official {PINNED_ZIP_NAME} (byte-identical)")
    ap.add_argument("--no-zip", action="store_true", help="stage only")
    args = ap.parse_args()

    # --- source facts ------------------------------------------------------
    launcher = DIST / EXES[0]
    bench = DIST / EXES[1]
    engine_dir = DIST / "engine"
    for need in (launcher, bench, engine_dir / "llama-server.exe",
                 engine_dir / "llama-cli.exe"):
        if not need.exists():
            print(f"[FAIL] missing build artifact: {need}")
            return 1
    launch_src = SRC / "cpulab_launch.py"
    bench_src = SRC / "cpulab_bench.py"
    version = read_kv(launch_src, "APP_VERSION") or "0.0.0"
    bench_version = read_kv(bench_src, "APP_VERSION") or "?"
    engine_build = read_kv(launch_src, "PINNED_BUILD") or "?"
    pinned_sha = read_kv(launch_src, "PINNED_SHA256") or "?"

    models_json = json.loads((SRC / "models.json").read_text(encoding="utf-8"))
    presets = models_json.get("presets", [])

    provenance = "official release zip (SHA256 verified at build time)"
    if args.verify_engine:
        problems = verify_engine_zip(engine_dir, Path(args.verify_engine), pinned_sha)
        if problems:
            print("[FAIL] engine provenance:")
            for p in problems:
                print("   -", p)
            return 1
        provenance = (f"byte-identical to official {PINNED_ZIP_NAME} "
                      f"(verified {date.today().isoformat()})")
        print(f"[OK  ] engine provenance: {provenance}")
    else:
        print("[note] engine provenance NOT re-verified this run "
              "(use --verify-engine; last verified 2026-10-23)")

    # --- staging -----------------------------------------------------------
    name = f"CPU-AI-Lab-{version}-win-x64"
    stage = STAGE_ROOT / name
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    stage.mkdir(parents=True)

    for exe in EXES:
        shutil.copy2(DIST / exe, stage / exe)
    shutil.copy2(SRC / "models.json", stage / "models.json")   # editable override
    shutil.copytree(engine_dir, stage / "engine")
    # llama.cpp MIT travels with the binaries (upstream zip ships only LLVM-OpenMP)
    shutil.copy2(PACK / "llamacpp-LICENSE.txt", stage / "engine" / "LICENSE")
    shutil.copy2(PACK / "LICENSE.txt", stage / "LICENSE.txt")
    (stage / "models").mkdir()
    shutil.copy2(PACK / "models-README.txt", stage / "models" / "README.txt")

    guides_src = PACK / "guides"
    guides_dst = stage / "guides"
    shutil.copytree(guides_src, guides_dst)

    # offline bookshelf (D40): reader ships under guides/ — minus authoring
    # tooling (build.bat, CONTENT-GUIDE.md) and promotion shots (Bookshelf_*.jpg
    # = landing-page assets, live in marketing/ only)
    books_src = ROOT / "books_packaged"
    if books_src.is_dir():
        shutil.copytree(books_src, guides_dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(
                            "build.bat", "CONTENT-GUIDE.md", "Bookshelf_*.jpg"))
        n_books = len(list((books_src / "books").rglob("*.md")))
        print(f"[OK  ] books: {n_books} titles from books_packaged/")
    else:
        n_books = 0
        print("[warn] books_packaged/ missing - zip would ship WITHOUT books")

    # guides = the bookshelf (D41: books supersede standalone guide files)
    staged_books = (len(list((guides_dst / "books").rglob("*.md")))
                    if (guides_dst / "books").is_dir() else 0)
    if not (guides_dst / "index.html").exists() or staged_books < 6:
        print(f"[warn] bookshelf incomplete: need index.html + 6 books "
              f"(found {staged_books} books)")

    # --- generated manifests ----------------------------------------------
    (stage / "VERSION.json").write_text(json.dumps({
        "product": "CPU AI Lab - Launcher + Honest Bench",
        "version": version,
        "bench_version": bench_version,
        "engine": {
            "build": engine_build,
            "source_zip": PINNED_ZIP_NAME,
            "zip_sha256": pinned_sha,
            "provenance": provenance,
        },
        "presets": len(presets),
        "books": n_books,
        "models_bundled": False,
        "package_date": date.today().isoformat(),
        "requirements": "Windows 10/11 x64; Vulkan GPU recommended (CPU-only works)",
    }, indent=2) + "\n", encoding="utf-8")

    sums = []
    for p in sorted(stage.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel = p.relative_to(stage).as_posix()
            sums.append(f"{sha256_file(p)}  {rel}")
    # newline="" keeps the file LF: sha256sum -c rejects CRLF checksum files
    (stage / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n",
                                          encoding="utf-8", newline="")
    print(f"[OK  ] staged {name}: {len(sums)} files, {len(presets)} presets")

    # --- frozen selftest FROM THE STAGING (acceptance) ---------------------
    proc = subprocess.run([str(stage / EXES[0]), "--selftest"],
                          capture_output=True, text=True, timeout=120,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or "RESULT: PASS" not in out:
        print("[FAIL] staged launcher selftest:")
        print(out)
        return 1
    print("[OK  ] staged launcher --selftest: PASS")

    # --- zip ---------------------------------------------------------------
    if args.no_zip:
        print(f"[done] stage only: {stage}")
        return 0
    zip_path = DIST / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, (Path(name) / p.relative_to(stage)).as_posix())
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[OK  ] {zip_path.name}  {size_mb:.1f} MB"
          + ("" if size_mb < 400 else "  <- check host size limits"))
    print(f"[done] deliverable ready: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
