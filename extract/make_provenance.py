"""Consolidate the per-analysis meta JSONs into a single provenance.json."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

META = {
    "tdr": DATA / "tdr" / "tdr_meta.json",
    "pca": DATA / "pca" / "pca_meta.json",
    "decoding": DATA / "decoding" / "decoding_meta.json",
    "decoding_expert": DATA / "decoding" / "decoding_expert_meta.json",
    "population_means": DATA / "population" / "population_means_meta.json",
    "population_by_projection": DATA / "population" / "population_means_by_projection_meta.json",
    "single_trial_heatmaps": DATA / "single_trial" / "single_trial_meta.json",
    "lick": DATA / "lick" / "lick_meta.json",
    "population_expert": DATA / "population" / "population_means_expert_meta.json",
    "decoding_timepooled": DATA / "decoding" / "decoding_timepooled_meta.json",
    "subgroups_selectivity": DATA / "subgroups" / "selectivity_meta.json",
    "subgroups_responsiveness_ttest": DATA / "subgroups" / "responsiveness_ttest_meta.json",
    "dimensionality": DATA / "dimensionality" / "dimensionality_meta.json",
}


def main() -> None:
    out = {
        "bundle": "results/transfer",
        "purpose": "Figure data + reproducible figure code for the main results.",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "canonical_record": "analysis/CANONICAL_RESULTS.md",
        "time_convention": "seconds, stimulus onset at t=0, stim window 0-2 s",
        "sessions_note": "13 first-reversal sessions, SAT016-SAT037 (see per-analysis meta).",
        "analyses": {},
    }
    for name, path in META.items():
        out["analyses"][name] = (json.loads(path.read_text()) if path.exists()
                                 else {"error": f"missing {path.name}"})
    (ROOT / "provenance.json").write_text(json.dumps(out, indent=2))
    print("Wrote", ROOT / "provenance.json")


if __name__ == "__main__":
    main()
