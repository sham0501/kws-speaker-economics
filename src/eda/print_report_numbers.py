#!/usr/bin/env python3
"""
One-off helper: reads the phase-1 JSON outputs and prints every number the
report template still needs, in a copy-paste-ready format.

Usage:
    python src/eda/print_report_numbers.py
Then copy the full terminal output and send it back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import resolve  # noqa: E402


def load(name: str) -> dict:
    p = resolve(f"reports/phase1/{name}")
    if not p.exists():
        print(f"[missing] {p} — run the matching script first.")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    audit = load("audit_summary.json")
    subset = load("subset_summary.json")
    eda = load("eda_manifest.json")

    print("=" * 70)
    print("COPY EVERYTHING BELOW THIS LINE AND SEND IT BACK")
    print("=" * 70)

    if audit:
        d = audit["duration_seconds"]
        print(f"duration.fraction_exactly_1s = {100 * (d.get('fraction_exactly_1s') or 0):.2f}%")
        a = audit["amplitude"]
        print(f"amplitude.peak_abs_median = {a.get('peak_abs_median')}")
        t = audit["table_level_issues"]
        print(f"table.missing_values_by_column = {t.get('missing_values_by_column')}")
        print(f"table.duplicate_filepath_rows = {t.get('duplicate_filepath_rows')}")
        print(f"table.duplicate_full_rows = {t.get('duplicate_full_rows')}")
        print(f"table.rows_with_invalid_utterance_index = {t.get('rows_with_invalid_utterance_index')}")
        print(f"flags.flag_corrupt = {audit['flags'].get('flag_corrupt')}")
        print(f"flags.flag_all_zero = {audit['flags'].get('flag_all_zero')}")
        print(f"low_energy_by_label = {audit.get('low_energy_by_label')}")
        for split, st in audit.get("speaker_stats_by_split", {}).items():
            print(
                f"speaker[{split}]: p25={st['samples_per_speaker_p25']:.1f} "
                f"p75={st['samples_per_speaker_p75']:.1f}"
            )

    if subset:
        print(f"silence_noise_file_assignment = {subset.get('silence_noise_file_assignment')}")

    if eda:
        print(f"figures_generated = {eda.get('figures')}")

    # PCA explained variance is inside the .npz cache, not a JSON file.
    npz_path = resolve("reports/phase1/speaker_profiles_phase1.npz")
    if npz_path.exists():
        import numpy as np

        z = np.load(npz_path)
        var = z["explained_variance_ratio"]
        print(f"pca.explained_variance_ratio (first 5) = {[round(float(v), 4) for v in var[:5]]}")
    else:
        print("pca.explained_variance_ratio: speaker_profiles_phase1.npz not found")

    print("=" * 70)


if __name__ == "__main__":
    main()
