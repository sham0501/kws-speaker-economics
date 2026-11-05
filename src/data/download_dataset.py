#!/usr/bin/env python3
"""
Download and extract the official Google Speech Commands v0.02 dataset.

Official source: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
The archive is roughly 2.3 GB; extracted, it's roughly 3.8 GB.

The raw data is never committed (data/raw is gitignored), so this script is
the record of how it was obtained: it writes a manifest with the URL, size,
SHA256 and download time to reports/phase1/download_manifest.json, which is
what makes the dataset rebuildable from the repository alone.

Usage (from the repo root):
    python src/data/download_dataset.py
    python src/data/download_dataset.py --skip-checksum   # quick smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import ensure_dir, load_config, resolve  # noqa: E402

CHUNK = 1024 * 256


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def download(url: str, dest: Path) -> None:
    """Download with progress reporting and resume support for interrupted transfers."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "kws-speaker-economics/1.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"[download] resuming from {_human(existing)} ...")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # If the server doesn't support Range, start over.
            if existing and resp.status != 206:
                print("[download] server didn't accept resume; starting over.")
                existing = 0
                tmp.unlink(missing_ok=True)

            total = int(resp.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing else "wb"
            done = existing
            t0 = time.time()
            with open(tmp, mode) as f:
                while True:
                    buf = resp.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total:
                        pct = 100.0 * done / total
                        speed = done / max(time.time() - t0, 1e-6) / 1e6
                        print(
                            f"\r[download] {pct:5.1f}%  {_human(done)} / {_human(total)}  "
                            f"({speed:.1f} MB/s)",
                            end="",
                            flush=True,
                        )
            print()
    except Exception as e:  # noqa: BLE001
        print(f"\n[download] error: {e}", file=sys.stderr)
        print(
            "[download] the partial file was kept; re-running the script will resume.",
            file=sys.stderr,
        )
        raise

    tmp.replace(dest)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
            done += len(buf)
            print(f"\r[checksum] {100.0 * done / size:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()


def safe_extract(tar_path: Path, out_dir: Path) -> None:
    """
    Safely extract the archive: paths outside out_dir (path traversal) are
    rejected. The official Speech Commands archive has no top-level
    folder, so it's extracted directly into out_dir.
    """
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = (out_dir / m.name).resolve()
            if not str(target).startswith(str(out_dir)):
                raise RuntimeError(f"suspicious entry in archive (path traversal): {m.name}")
        total = len(members)
        for i, m in enumerate(members, 1):
            tar.extract(m, out_dir)
            if i % 2000 == 0 or i == total:
                print(f"\r[extract] {i}/{total} files", end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Speech Commands v0.02")
    ap.add_argument("--config", default=None, help="path to config file (default configs/phase1.yaml)")
    ap.add_argument("--skip-checksum", action="store_true", help="skip SHA256 verification")
    ap.add_argument("--keep-archive", action="store_true", help="keep the tar.gz after extraction")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]

    raw_dir = ensure_dir(ds["raw_dir"])
    archive = raw_dir / "speech_commands_v0.02.tar.gz"
    extract_dir = raw_dir / ds["extract_dirname"]

    # Don't redo the work if it's already extracted.
    if (extract_dir / "validation_list.txt").exists() and (extract_dir / "testing_list.txt").exists():
        n_wav = sum(1 for _ in extract_dir.rglob("*.wav"))
        print(f"[skip] dataset already present at {extract_dir} ({n_wav} wav files).")
        return 0

    if not archive.exists():
        print(f"[download] starting download from {ds['url']}")
        print("[download] note: the archive is roughly 2.3 GB.")
        download(ds["url"], archive)
    else:
        print(f"[skip] archive already present: {archive} ({_human(archive.stat().st_size)})")

    digest = None
    if not args.skip_checksum:
        print("[checksum] computing SHA256 ...")
        digest = sha256_of(archive)
        expected = ds.get("sha256")
        if expected and digest != expected:
            print(
                f"[checksum] warning: SHA256 mismatch.\n"
                f"           expected:  {expected}\n"
                f"           computed:  {digest}\n"
                f"           this is expected if you downloaded from a different mirror; the\n"
                f"           actual value is recorded in the manifest. Otherwise, delete the\n"
                f"           file and download again.",
                file=sys.stderr,
            )
        else:
            print(f"[checksum] verified: {digest}")

    print(f"[extract] extracting to {extract_dir} ...")
    safe_extract(archive, extract_dir)

    n_wav = sum(1 for _ in extract_dir.rglob("*.wav"))
    labels = sorted(p.name for p in extract_dir.iterdir() if p.is_dir())

    manifest = {
        "url": ds["url"],
        "archive_bytes": archive.stat().st_size,
        "sha256": digest,
        "sha256_expected": ds.get("sha256"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "extract_dir": str(extract_dir.relative_to(resolve("."))),
        "n_wav_files": n_wav,
        "label_dirs": labels,
        "has_validation_list": (extract_dir / "validation_list.txt").exists(),
        "has_testing_list": (extract_dir / "testing_list.txt").exists(),
    }
    out = ensure_dir(cfg["paths"]["reports_dir"]) / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] extracted {n_wav} wav files across {len(labels)} folders.")
    print(f"[done] manifest written to: {out}")

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print("[cleanup] removed the tar.gz archive (pass --keep-archive to keep it).")

    free = shutil.disk_usage(raw_dir).free
    print(f"[info] free disk space remaining: {_human(free)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
