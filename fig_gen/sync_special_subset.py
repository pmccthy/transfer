#!/usr/bin/env python3
"""Copy the hand-picked "favourite" figures into figs/special_subset/, renamed
without the FIGX/MODELFIGX-style numeric tags -- named by data source instead:

    neuraldata_...   real experimental data only  (neuronal-representations)
    RNN_...          RNN model data only, no real-data comparison
    comparison_...   RNN model vs. real experimental data (chi2 fits etc.)

Reads from data/{combined,experimental}/ (the mirror sync_combined.sh /
sync_experimental.sh just refreshed), so run this AFTER those, not instead of
them -- sync_all.sh already does this (it's the last step). Re-run any time
after regenerating figures to refresh the copies in place.

To add a figure to the special subset: pick its category (neuraldata / RNN /
comparison), find its current path under data/combined/figs/composites/ or
data/experimental/figures/figs/composites/ (strip the trailing .png/.pdf),
and add a (category, that_path, new_basename) row to SPECIAL_SUBSET below.
new_basename is what comes after the category prefix -- reuse the descriptive
part of the original filename (after the tag's "__") so the name stays
self-explanatory.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "figs" / "special_subset"

# (category, path relative to data/ with no extension, new descriptive basename)
SPECIAL_SUBSET = [
    ("neuraldata",
     "experimental/figures/figs/composites/FIG2__reversal_decoding_lettered_nolegend",
     "reversal_decoding_lettered_nolegend"),
    ("comparison",
     "combined/figs/composites/time_averaged/MODELFIG2_time_averaged__crossmodel_summary_5k_lettered_nolegend_time_averaged",
     "crossmodel_summary_5k_lettered_nolegend_time_averaged"),
    ("RNN",
     "combined/figs/composites/time_averaged/coarse/MODELFIG1_POSTREV_time_averaged__model_postreversal_dynamics_5k_lettered_nolegend_time_averaged",
     "postreversal_dynamics_5k_lettered_nolegend_time_averaged"),
    ("RNN",
     "combined/figs/composites/time_averaged/coarse/MODELFIG1_EXPERT_time_averaged__model_expert_summary_5k_lettered_nolegend_time_averaged",
     "expert_summary_5k_lettered_nolegend_time_averaged"),
    ("neuraldata",
     "experimental/figures/figs/composites/time_averaged/FIG1__expert_reversal_overview_responders_lettered_nolegend_time_averaged",
     "expert_reversal_overview_responders_lettered_nolegend_time_averaged"),
    ("RNN",
     "combined/figs/composites/MECHFIG2__mechanistic_postreversal_summary_5k_lettered_nolegend",
     "mechanistic_postreversal_summary_5k_lettered_nolegend"),
    ("RNN",
     "combined/figs/composites/NOISEFIG__noise_robustness_lettered_nolegend",
     "noise_robustness_lettered_nolegend"),
    ("RNN",
     "combined/figs/composites/PCAFIG3D__pca_phase_space_3d_5k_lettered",
     "pca_phase_space_3d_5k_lettered"),
    ("RNN",
     "combined/figs/composites/TDRFIG3D__tdr_phase_space_3d_5k_lettered",
     "tdr_phase_space_3d_5k_lettered"),
    ("RNN",
     "combined/figs/composites/RNN_PCA_SEEDS_ALL__pca_individual_seeds_all_models_5k",
     "pca_individual_seeds_all_models_5k"),
    ("RNN",
     "combined/figs/composites/RNN_TDR_SEEDS_ALL__tdr_individual_seeds_all_models_5k",
     "tdr_individual_seeds_all_models_5k"),
    ("RNN",
     "combined/figs/composites/RNN_PCA_SEEDS_ALL_3D__pca_individual_seeds_all_models_3d_5k",
     "pca_individual_seeds_all_models_3d_5k"),
    ("RNN",
     "combined/figs/composites/RNN_TDR_SEEDS_ALL_3D__tdr_individual_seeds_all_models_3d_5k",
     "tdr_individual_seeds_all_models_3d_5k"),
    ("neuraldata",
     "experimental/figures/figs/composites/PCAFIG3D_EXP__pca_phase_space_3d_combined",
     "pca_phase_space_3d_combined"),
    ("neuraldata",
     "experimental/figures/figs/composites/TDRFIG3D_EXP__tdr_phase_space_3d_combined",
     "tdr_phase_space_3d_combined"),
    ("neuraldata",
     "experimental/figures/figs/composites/EXP_PCA_SESSIONS__pca_individual_sessions_n10",
     "pca_individual_sessions_n10"),
    ("neuraldata",
     "experimental/figures/figs/composites/EXP_TDR_SESSIONS__tdr_individual_sessions_n10",
     "tdr_individual_sessions_n10"),
    ("neuraldata",
     "experimental/figures/figs/composites/EXP_PCA_SESSIONS_3D__pca_individual_sessions_3d_n10",
     "pca_individual_sessions_3d_n10"),
    ("neuraldata",
     "experimental/figures/figs/composites/EXP_TDR_SESSIONS_3D__tdr_individual_sessions_3d_n10",
     "tdr_individual_sessions_3d_n10"),
    ("neuraldata",
     "experimental/figures/figs/composites/POPPROJCMP__projection_status_comparison_lettered",
     "projection_status_comparison_lettered"),
    ("neuraldata",
     "experimental/figures/figs/composites/time_averaged/FIG1VIG__expert_reversal_overview_vigourchange_responders_lettered_nolegend_time_averaged",
     "expert_reversal_overview_vigourchange_responders_lettered_nolegend_time_averaged"),
    ("comparison",
     "combined/figs/composites/GROUPSIZES_MODEL__group_sizes_by_model_lettered_responders_time_averaged",
     "group_sizes_by_model_lettered_responders_time_averaged"),
]

EXTS = (".pdf",)  # the user has kept vector copies only in special_subset so far


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = 0
    for category, rel_path, new_basename in SPECIAL_SUBSET:
        src_stem = DATA / rel_path
        dst_stem = OUT / f"{category}_{new_basename}"
        for ext in EXTS:
            src = src_stem.with_suffix(ext)
            dst = dst_stem.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  {dst.name}  <-  data/{rel_path}{ext}")
                n_ok += 1
            else:
                print(f"  ** MISSING source, skipped: data/{rel_path}{ext}", file=sys.stderr)
                n_missing += 1
    print(f"\nspecial_subset: {n_ok} file(s) refreshed" + (f", {n_missing} missing" if n_missing else ""))
    if n_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
