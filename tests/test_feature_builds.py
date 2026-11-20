"""
A --limit smoke test must not replace the full feature build, and loaders
must refuse a build that does not match the current manifest or settings.

Needs a full build first (python src/features/build_features.py); this test
then runs a small limited build itself.

    python tests/test_feature_builds.py
"""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.store import StaleFeatures, load_features, load_waveforms  # noqa: E402
from utils.common import load_config, resolve  # noqa: E402

CFG = load_config("configs/phase2.yaml")
FULL = resolve(CFG["paths"]["processed_dir"])
REPORTS = resolve(CFG["paths"]["reports_dir"])


def digest(paths):
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def test_limited_run_leaves_full_build_untouched():
    watched = [FULL / n for n in ("clip_features.npy", "clip_lengths.npy", "features_meta.json")]
    watched += [REPORTS / "features_summary.json"]
    before = digest(watched)
    subprocess.run([sys.executable, str(ROOT / "src/features/build_features.py"), "--limit", "120", "--jobs", "1"],
                   check=True, cwd=ROOT, capture_output=True)
    assert digest(watched) == before
    man, feats, _ = load_features(CFG, limited=True)
    assert len(man) == len(feats) and len(man) >= 36
    assert set(man["split"]) == {"train", "val", "test"} and man["label"].nunique() == 12
    full_man, full_feats, _ = load_features(CFG)
    assert len(full_man) == len(full_feats) == len(load_waveforms(CFG))


def test_changed_settings_are_refused():
    cfg = copy.deepcopy(CFG)
    cfg["features"]["n_mels"] = 64
    try:
        load_features(cfg)
    except StaleFeatures:
        return
    raise AssertionError("features built with other settings were accepted")


def test_old_extractor_is_refused():
    meta_path = FULL / "features_meta.json"
    original = meta_path.read_text(encoding="utf-8")
    meta = json.loads(original)
    meta["settings_sha1"] = "built-by-an-older-version"
    try:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        load_features(CFG)
    except StaleFeatures:
        return
    finally:
        meta_path.write_text(original, encoding="utf-8")
    raise AssertionError("features from an older extractor were accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
