"""Projection status + projection-resolved per-condition population means.

Some neurons project to striatum and some don't. This status is recorded in the
same ``*_snr_filtered_cells.csv`` files as the SNR filter, in the columns
``is_projecting`` / ``is_nonprojecting`` (``cellID`` is 0-based and matches the
cell dimension of the TDR ``cond_avg`` arrays via ``filtered_cell_indices``).

This extractor:
  * writes a per-cell projection-status table, and
  * splits the per-condition population mean (from the TDR ``cond_avg``, same
    baseline-subtracted dF/F used by the plain population means) into
    projecting vs non-projecting neurons.

Outputs (results/transfer/data/population/):
  projection_status.csv                    session, filt_index, cellID, projection_status
  population_means_by_projection.csv       condition, projection_status, frame, time_s,
                                           pop_mean_pooled, sem_over_sessions,
                                           n_neurons_total, n_sessions
  population_means_by_projection_meta.json provenance / cell counts
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sources import TDR_DIR, snr_session_dir, DATA_ROOT, require  # noqa: E402

OUT = DATA_ROOT / "population"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = ["projecting", "nonprojecting"]   # main comparison (unclassified tracked separately)


def _projection_group(row):
    if row.get("is_projecting", 0) == 1:
        return "projecting"
    if row.get("is_nonprojecting", 0) == 1:
        return "nonprojecting"
    return "unclassified"


def main() -> None:
    require(TDR_DIR, "Canonical TDR run (cond_avg + filtered_cell_indices)")
    summary = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())
    names = summary["session_names"]

    status_rows = []
    trial_type_list = None
    t_ax = None
    # accumulators per group: pooled sum, cell count, list of per-session means
    sum_pooled = {g: None for g in GROUPS}
    n_cells = {g: 0 for g in GROUPS}
    sess_means = {g: [] for g in GROUPS}
    counts_per_session = {g: [] for g in GROUPS}

    for name in names:
        tdr = pickle.load(open(TDR_DIR / f"{name}_tdr_v2.pkl", "rb"))
        cond_avg = np.asarray(tdr["cond_avg"])          # (n_cond, n_filt, n_time)
        fci = np.asarray(tdr["filtered_cell_indices"]).astype(int)
        ttl = list(tdr["trial_type_list"])
        if trial_type_list is None:
            trial_type_list = ttl
            t_ax = np.asarray(tdr["t_ax"])
            for g in GROUPS:
                sum_pooled[g] = np.zeros((cond_avg.shape[0], cond_avg.shape[2]))

        csvs = glob.glob(f"{snr_session_dir(name)}/*_snr_filtered_cells.csv")
        if not csvs:
            print(f"  ! {name}: no SNR/projection CSV — skipping")
            continue
        proj = pd.read_csv(csvs[0]).set_index("cellID")
        sub = proj.loc[fci]
        groups = sub.apply(_projection_group, axis=1).values   # per filtered cell

        for i, (cid, grp) in enumerate(zip(fci, groups)):
            status_rows.append((name, i, int(cid), grp))

        for g in GROUPS:
            mask = groups == g
            if mask.sum() == 0:
                continue
            grp_mean = cond_avg[:, mask, :].mean(axis=1)        # (n_cond, n_time)
            sum_pooled[g] += cond_avg[:, mask, :].sum(axis=1)
            n_cells[g] += int(mask.sum())
            sess_means[g].append(grp_mean)
            counts_per_session[g].append(int(mask.sum()))
        print(f"  ✓ {name}: " + ", ".join(f"{g}={int((groups==g).sum())}" for g in GROUPS))

    # per-cell status table
    pd.DataFrame(status_rows, columns=["session", "filt_index", "cellID",
                                       "projection_status"]
                 ).to_csv(OUT / "projection_status.csv", index=False)

    # combined projection-resolved population means
    rows = []
    for g in GROUPS:
        if n_cells[g] == 0:
            continue
        pooled = sum_pooled[g] / n_cells[g]                    # (n_cond, n_time)
        stack = np.stack(sess_means[g], axis=0)                # (n_sess_g, n_cond, n_time)
        sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 \
            else np.zeros_like(pooled)
        for ci, cond in enumerate(trial_type_list):
            for ti in range(pooled.shape[1]):
                rows.append((cond, g, ti, float(t_ax[ti]), float(pooled[ci, ti]),
                             float(sem[ci, ti]), n_cells[g], stack.shape[0]))
    pd.DataFrame(rows, columns=["condition", "projection_status", "frame", "time_s",
                                "pop_mean_pooled", "sem_over_sessions",
                                "n_neurons_total", "n_sessions"]
                 ).to_csv(OUT / "population_means_by_projection.csv", index=False)

    meta = {
        "analysis": "Per-condition population means split by projection status",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(TDR_DIR),
        "projection_source": "*_snr_filtered_cells.csv (is_projecting / is_nonprojecting), cellID 0-based",
        "signal": "TDR cond_avg (baseline-subtracted dF/F), averaged over neurons per group",
        "groups": GROUPS,
        "unclassified_excluded_from_groups": True,
        "n_projecting_total": n_cells["projecting"],
        "n_nonprojecting_total": n_cells["nonprojecting"],
        "trial_type_list": trial_type_list,
        "n_sessions": len(names),
        "notes": "projecting = projects to striatum (is_projecting==1). "
                 "pop_mean_pooled = neuron-weighted mean over that group pooled across "
                 "sessions; sem_over_sessions across the per-session group means.",
    }
    (OUT / "population_means_by_projection_meta.json").write_text(json.dumps(meta, indent=2))
    print("Projection-resolved population extraction complete →", OUT)


if __name__ == "__main__":
    main()
