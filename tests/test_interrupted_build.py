"""
A feature build that is interrupted must not leave arrays the loaders accept.

Runs entirely in a temporary folder: a small build is completed, a second
build into the same folder is killed while it is writing, and the loader
must then refuse the folder instead of reading a half-written array.

    python tests/test_interrupted_build.py      (or: python -m pytest tests)
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.store import load_features, load_waveforms  # noqa: E402
from utils.common import load_config  # noqa: E402

N = 2400  # large enough that the second build can be stopped half-way


def build(cfg_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(ROOT / "src/features/build_features.py"), "--config", str(cfg_path), "--limit", str(N), "--jobs", "1"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def test_interrupted_build_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = load_config("configs/phase2.yaml")
        cfg["paths"]["processed_dir"] = str(tmp / "processed")
        cfg["paths"]["reports_dir"] = str(tmp / "reports")
        cfg_path = tmp / "phase2.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        assert build(cfg_path).wait() == 0
        load_features(cfg, limited=True)  # a finished build is accepted

        waves = tmp / "processed" / "limited" / "waveforms.npy"
        first = waves.stat().st_mtime_ns
        proc = build(cfg_path)
        deadline = time.time() + 60
        # wait until the second build has started rewriting the waveform array
        while waves.stat().st_mtime_ns == first and proc.poll() is None and time.time() < deadline:
            time.sleep(0.005)
        time.sleep(0.5)  # let it write part of the new arrays
        assert proc.poll() is None, "the build finished before it could be interrupted; raise N"
        proc.kill()
        proc.wait()

        for loader in (load_features, load_waveforms):
            try:
                loader(cfg, limited=True)
            except FileNotFoundError:
                continue
            raise AssertionError(f"{loader.__name__} accepted an interrupted build")


if __name__ == "__main__":
    test_interrupted_build_is_refused()
    print("ok  test_interrupted_build_is_refused")
