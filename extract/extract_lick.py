"""Extract canonical per-trial anticipatory lick rate → tidy CSV.

Reproduces the blessed computation::

    beh_df   = pd.read_csv(recordingList.behFileName[ind], header=0, index_col=0)
    lickData = pd.read_pickle(recordingList.lickFileName[ind])
    fLickRate = lickData['sample_rate']
    rig = recordingList.rig[ind]
    lickrates, beh_df = mfun.lickrate_per_trial(
        beh_df, lickData, rig, deltalick=False,
        baselinelick_s=2, testlick_s=2, fLickRate=fLickRate, bout=True)

``lickrates`` is the stimulus-evoked (anticipatory) lick rate per trial in the
2 s window after stimulus onset, bout-corrected (from first lick post-stim).

The session's ``recordingList`` (with behFileName / lickFileName / rig) is read
directly from the loader pkl, so this does not need ``matlab`` / ``mfun.analysis()``.

REQUIREMENTS (present in the analysis environment, not in this sandbox):
  * the ``cingulatedms`` + ``LakLabAnalysis`` packages (TRANSFER_CINGULATE_ROOT),
  * the raw behaviour + lick files referenced in recordingList (acquisition
    volume, e.g. /Volumes/STan/...). Use TRANSFER_LICK_PATH_FROM/TO to remap if
    your copy lives elsewhere.

Output (results/transfer/data/lick/):
  lick_rates_per_trial.csv   session, trial_number, stimulus_label,
                             stimulus_type, reward_prob_rev, anticipatory_lick_rate
  lick_meta.json             provenance / parameters

Schema note: ``plot_lick.py`` consumes lick_rates_per_trial.csv. If you compute
lick rates another way, just produce a CSV with those columns.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _sources import (TDR_DIR, CINGULATE_ROOT, DATA_ROOT, loader_pkl_path,  # noqa: E402
                      remap_data_path, require)
from _loader_pkl import load_loader  # noqa: E402

import os  # noqa: E402

OUT = DATA_ROOT / "lick"
OUT.mkdir(parents=True, exist_ok=True)
_REPO_ROOT = Path(__file__).resolve().parents[3]

LICK_PARAMS = dict(deltalick=False, baselinelick_s=2, testlick_s=2, bout=True)


def resolve_sessions():
    """Pick which sessions to extract licks for and the output file suffix.

    Controlled by env vars:
      TRANSFER_LICK_SESSION_SET = 'reversal' (default) | 'expert'
      TRANSFER_LICK_SESSIONS    = path to a JSON list of session names (overrides)
    Returns (session_names, out_suffix).
    """
    custom = os.environ.get("TRANSFER_LICK_SESSIONS")
    if custom and Path(custom).exists():
        data = json.loads(Path(custom).read_text())
        names = data if isinstance(data, list) else data.get("sessions", list(data))
        return names, os.environ.get("TRANSFER_LICK_SUFFIX", "_custom")

    which = os.environ.get("TRANSFER_LICK_SESSION_SET", "reversal").lower()
    if which == "expert":
        p = _REPO_ROOT / "metadata" / "SAT_good_expert_sessions.json"
        data = json.loads(p.read_text())
        names = data["sessions"] if isinstance(data, dict) else data
        return names, "_expert"
    # default: the 13 reversal sessions (canonical TDR run)
    names = json.loads((TDR_DIR / "analysis_summary_v2.json").read_text())["session_names"]
    return names, ""


def _label_reversal(reward_prob_rev: str) -> str:
    """Reversal cue identity from rewardProbRev (e.g. '100% Rewarded-->0% ...')."""
    s = str(reward_prob_rev)
    if s.startswith("100%"):
        return "100_to_0"
    if s.startswith("0%"):
        return "0_to_100"
    return "50"


def _label_expert(reward_prob: str) -> str:
    """Fixed reward-probability stimulus from rewardProb (e.g. '100% Rewarded')."""
    s = str(reward_prob)
    if s.startswith("100%"):
        return "100"
    if s.startswith("0%"):
        return "0"
    return "50"


def main() -> None:
    require(CINGULATE_ROOT, "cingulateDMS repo (TRANSFER_CINGULATE_ROOT)")
    sys.path.insert(0, str(CINGULATE_ROOT))
    try:
        import cingulatedms.main_funcs as mfun
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import cingulatedms.main_funcs — its dependencies "
            "(tifffile, LakLabAnalysis, ...) must be installed in this "
            f"environment. Original error: {exc}"
        )

    session_names, suffix = resolve_sessions()
    print(f"Extracting licks for {len(session_names)} sessions (suffix='{suffix or 'reversal'}')")

    rows = []
    for name in session_names:
        try:
            L = load_loader(loader_pkl_path(name))
        except FileNotFoundError:
            print(f"  ! {name}: loader pkl missing — skipping")
            continue
        rl = L.info.recordingList
        col = "blockName" if "blockName" in rl.columns else "sessionName"
        m = rl[rl[col].astype(str) == name]
        if len(m) == 0:
            print(f"  ! {name}: not found in recordingList — skipping")
            continue
        ind = m.index[0]
        beh_path = remap_data_path(rl["behFileName"][ind])
        lick_path = remap_data_path(rl["lickFileName"][ind])
        rig = rl["rig"][ind]

        beh_df = pd.read_csv(beh_path, header=0, index_col=0)
        lick_data = pd.read_pickle(lick_path)
        f_lick = lick_data["sample_rate"] if "sample_rate" in lick_data else 2000
        lickrates, beh_df = mfun.lickrate_per_trial(
            beh_df, lick_data, rig, fLickRate=f_lick, **LICK_PARAMS)

        lickrates = np.asarray(lickrates, dtype=float)
        stim_type = beh_df["stimulusType"].values
        # Reversal sessions carry 'rewardProbRev' (cue identity across reversal);
        # expert sessions carry only 'rewardProb' (fixed reward probability).
        if "rewardProbRev" in beh_df.columns:
            src = beh_df["rewardProbRev"].values
            labels = [_label_reversal(x) for x in src]
        else:
            src = beh_df["rewardProb"].values
            labels = [_label_expert(x) for x in src]
        for i, lr in enumerate(lickrates):
            rows.append((name, i, labels[i], stim_type[i], src[i], float(lr)))
        print(f"  ✓ {name}: {len(lickrates)} trials")

    pd.DataFrame(rows, columns=["session", "trial_number", "stimulus_label",
                                "stimulus_type", "reward_prob_rev",
                                "anticipatory_lick_rate"]
                 ).to_csv(OUT / f"lick_rates_per_trial{suffix}.csv", index=False)

    meta = {
        "analysis": "Anticipatory (stimulus-evoked) lick rate per trial",
        "session_set": suffix.lstrip("_") or "reversal",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": "cingulatedms.main_funcs.lickrate_per_trial",
        "parameters": LICK_PARAMS,
        "definition": ("licks/second in the 2 s window after stimulus onset, "
                       "bout-corrected (measured from first lick post-stim); "
                       "deltalick=False returns the test-window rate (not baseline-subtracted)."),
        "n_sessions": len(set(r[0] for r in rows)),
        "sessions": sorted(set(r[0] for r in rows)),
        "stimulus_labels": {"100_to_0": "100%->0% reward prob stimulus",
                            "50": "50% stimulus", "0_to_100": "0%->100% stimulus"},
        "recording_list_source": "info.recordingList inside each loader pkl",
    }
    (OUT / f"lick_meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    print(f"Lick extraction complete → {OUT}/lick_rates_per_trial{suffix}.csv")


if __name__ == "__main__":
    main()
