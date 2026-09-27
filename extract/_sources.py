"""Source-run locations for the `transfer` extraction scripts.

Each extraction script reads a *blessed* result directory, pulls out the arrays
needed to redraw the canonical figures, and writes tidy CSV + JSON into
``results/transfer/data/``.

Paths default to the locations on the machine where the runs were produced
(``~/Documents/experimental_results/...``). Override any of them with an
environment variable if your copy lives elsewhere, e.g.::

    TRANSFER_TDR_DIR=/path/to/tdr python extract/extract_tdr.py

The canonical TDR run is recorded in ``analysis/CANONICAL_RESULTS.md``. The PCA,
decoding and population runs are the matching-preprocessing companions of that
run (SNR-filtered, pre-trial baseline-subtracted, 13 first-reversal sessions);
they are not yet individually blessed in CANONICAL_RESULTS.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# Root that holds all experimental_results run directories.
_EXP_ROOT = Path(
    os.environ.get(
        "TRANSFER_EXP_ROOT",
        str(Path.home() / "Documents" / "experimental_results"),
    )
)

# --- Canonical TDR (see analysis/CANONICAL_RESULTS.md) -----------------------
TDR_DIR = Path(
    os.environ.get(
        "TRANSFER_TDR_DIR",
        str(_EXP_ROOT
            / "08_04_26_reversal_dff_from_flu_processed_baseline_sub_snr_val_peak_2_3s_tdr_v2"),
    )
)

# --- PCA companion of the canonical TDR run ----------------------------------
PCA_DIR = Path(
    os.environ.get(
        "TRANSFER_PCA_DIR",
        str(_EXP_ROOT
            / "08_04_26_reversal_dff_from_flu_processed_baseline_sub_snr_pca"),
    )
)

# --- Decoding (SNR-filtered, pre-trial baseline-subtracted) ------------------
DECODING_DIR = Path(
    os.environ.get(
        "TRANSFER_DECODING_DIR",
        str(_EXP_ROOT / "06_04_26_local_decoding_rerun_snr_filtered_pretrial_baseline_sub"),
    )
)
DECODING_TIME_RESOLVED_PKL = (
    DECODING_DIR
    / "decoder_pipeline_outputs"
    / "time_resolved_decoding"
    / "time_resolved_decoding_results.pkl"
)

# --- Raw session loader pkls (single-trial dF/F + behaviour) ------------------
# Tree: <root>/<animalID>/<session>/<session>.pkl  (SAT_dff_from_flu_processed)
LOADER_ROOT = Path(
    os.environ.get(
        "TRANSFER_LOADER_ROOT",
        str(Path.home() / "Documents" / "experimental_data" / "SAT_dff_from_flu_processed"),
    )
)


def loader_pkl_path(session_name: str) -> Path:
    """Path to a session's loader pkl: <root>/<animalID>/<session>/<session>.pkl."""
    animal_id = session_name.split("_")[-1]
    return LOADER_ROOT / animal_id / session_name / f"{session_name}.pkl"


# --- SNR-filtered cell CSVs (1-based cellID; decoding uses cellID-1) ----------
SNR_ROOT = Path(
    os.environ.get(
        "TRANSFER_SNR_ROOT",
        str(Path.home() / "Documents" / "experimental_data" / "SAT"),
    )
)


def snr_session_dir(session_name: str) -> Path:
    """Directory holding a session's *_snr_filtered_cells.csv."""
    animal_id = session_name.split("_")[-1]
    return SNR_ROOT / animal_id / session_name


# --- cingulateDMS analysis code (for the canonical lick-rate computation) -----
# Repo root that contains the ``cingulatedms`` and ``LakLabAnalysis`` packages.
CINGULATE_ROOT = Path(
    os.environ.get(
        "TRANSFER_CINGULATE_ROOT",
        str(Path.home() / "Documents" / "cingulateDMS"),
    )
)

# Optional path remap for the raw behaviour/lick files recorded in the loader's
# recordingList (they point at the acquisition volume, e.g. /Volumes/STan/...).
# Set TRANSFER_LICK_PATH_FROM / TRANSFER_LICK_PATH_TO to rewrite that prefix.
LICK_PATH_FROM = os.environ.get("TRANSFER_LICK_PATH_FROM")
LICK_PATH_TO = os.environ.get("TRANSFER_LICK_PATH_TO")


def remap_data_path(p: str) -> str:
    """Apply the optional lick/behaviour path prefix remap."""
    if LICK_PATH_FROM and LICK_PATH_TO and isinstance(p, str) and p.startswith(LICK_PATH_FROM):
        return LICK_PATH_TO + p[len(LICK_PATH_FROM):]
    return p

# --- Output location ---------------------------------------------------------
TRANSFER_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = TRANSFER_ROOT / "data"


def require(path: Path, what: str) -> Path:
    """Raise a clear error if a source path is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at:\n    {path}\n"
            "Set the corresponding TRANSFER_*_DIR environment variable to point "
            "at your copy of the run (see extract/_sources.py)."
        )
    return path
