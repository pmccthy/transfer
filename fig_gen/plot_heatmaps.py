"""Neuron x condition response heatmaps from the extracted single-trial data.

Two data sources (both in ../data):
  * heatmap/heatmap_condition_means.csv  — the default full-session heatmap
    (SNR-filtered cells, evoked response averaged over ALL trials per condition).
  * single_trial/<session>_single_trial.npz + single_trial_metadata.csv — the
    trial-resolution store, used by ``condition_means_from_trials`` to rebuild a
    heatmap over ANY subset of trials (the intended future use).

Produces:
  heatmap_combined.png            — all SNR neurons (pooled across sessions) x condition
  heatmap_per_session.png         — one heatmap panel per session
  heatmap_trial_subset_demo.png   — combined heatmap for the first half vs second
                                    half of trials in each session, showing how to
                                    recompute from a trial subset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUTDIR, DATA

CONDITIONS = [
    "100%-->0% (pre-Rev)", "50% (pre-Rev)", "0%-->100% (pre-Rev)",
    "100%-->0% (post-Rev)", "50% (post-Rev)", "0%-->100% (post-Rev)",
]
COND_SHORT = [c.replace(" (pre-Rev)", "\npre").replace(" (post-Rev)", "\npost")
              for c in CONDITIONS]
CMAP = "RdBu_r"


# --------------------------------------------------------------------------- #
# Trial-resolution helper: rebuild a neuron x condition matrix from any subset
# --------------------------------------------------------------------------- #
def condition_means_from_trials(session, trial_mask=None, snr_only=True,
                                subtract_baseline=True):
    """Return (cells x n_conditions) evoked means for a session, over a subset.

    Args:
        session: session name (e.g. '2025-04-30_1_SAT016').
        trial_mask: boolean array over the session's trials, or a callable
            (session_trial_number, condition) -> bool, or None for all trials.
        snr_only: restrict to SNR-filtered cells.
        subtract_baseline: use resp-base (evoked) vs raw resp.

    Returns:
        (matrix (n_cells, n_cond), cell_index array, conditions list).
    """
    d = np.load(DATA / "single_trial" / f"{session}_single_trial.npz",
                allow_pickle=True)
    resp = d["resp_mean"].astype(float)
    base = d["base_mean"].astype(float)
    signal = resp - base if subtract_baseline else resp
    conds = d["trial_condition"].astype(str)
    tnum = d["trial_session_number"]
    cell_index = d["cell_index"]
    snr = d["snr_mask"]

    n_trials = signal.shape[1]
    if trial_mask is None:
        sel = np.ones(n_trials, dtype=bool)
    elif callable(trial_mask):
        sel = np.array([bool(trial_mask(tnum[i], conds[i])) for i in range(n_trials)])
    else:
        sel = np.asarray(trial_mask, dtype=bool)

    mat = np.full((signal.shape[0], len(CONDITIONS)), np.nan)
    for ci, cond in enumerate(CONDITIONS):
        col = sel & (conds == cond)
        if col.any():
            mat[:, ci] = signal[:, col].mean(axis=1)

    if snr_only:
        keep = snr.astype(bool)
        return mat[keep], cell_index[keep], CONDITIONS
    return mat, cell_index, CONDITIONS


# --------------------------------------------------------------------------- #
def _sort_rows(mat):
    """Sort neurons by preferred (argmax) condition, then by that peak value."""
    valid = ~np.all(np.isnan(mat), axis=1)
    m = np.nan_to_num(mat)
    pref = np.argmax(m, axis=1)
    peak = m[np.arange(len(m)), pref]
    order = np.lexsort((-peak, pref))
    order = order[valid[order]]
    return order


def _draw(ax, mat, title, vlim):
    order = _sort_rows(mat)
    im = ax.imshow(mat[order], aspect="auto", cmap=CMAP, vmin=-vlim, vmax=vlim,
                   interpolation="nearest")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(COND_SHORT, fontsize=7)
    ax.set_ylabel("neuron (sorted)")
    ax.set_title(title, fontsize=9)
    return im


def _matrix_from_csv():
    df = pd.read_csv(DATA / "heatmap" / "heatmap_condition_means.csv")
    df["uid"] = df.session + "#" + df.cell_index.astype(str)
    wide = df.pivot_table(index="uid", columns="condition", values="mean_evoked")
    wide = wide.reindex(columns=CONDITIONS)
    return df, wide


def plot_combined():
    _, wide = _matrix_from_csv()
    mat = wide.values
    vlim = np.nanpercentile(np.abs(mat), 98)
    fig, ax = plt.subplots(figsize=(5.2, 7))
    im = _draw(ax, mat, f"All SNR neurons x condition (n={mat.shape[0]})", vlim)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="evoked dF/F")
    fig.tight_layout()
    fig.savefig(OUTDIR / "heatmap_combined.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "heatmap_combined.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_per_session():
    df, _ = _matrix_from_csv()
    sessions = sorted(df.session.unique())
    ncols = 4
    nrows = -(-len(sessions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows),
                             squeeze=False)
    allvals = df.mean_evoked.values
    vlim = np.nanpercentile(np.abs(allvals), 98)
    im = None
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sub = df[df.session == sess].pivot_table(
            index="cell_index", columns="condition", values="mean_evoked"
        ).reindex(columns=CONDITIONS)
        im = _draw(ax, sub.values, sess.split("_")[-1] + " " + sess[:10], vlim)
    for j in range(len(sessions), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="evoked dF/F")
    fig.suptitle("Neuron x condition heatmaps by session (SNR-filtered)", fontsize=11)
    fig.savefig(OUTDIR / "heatmap_per_session.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "heatmap_per_session.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_trial_subset_demo():
    """Rebuild the combined heatmap for early vs late trials, via the npz store."""
    meta = pd.read_csv(DATA / "single_trial" / "single_trial_metadata.csv")
    halves = {"first half": [], "second half": []}
    for sess, g in meta.groupby("session"):
        med = g.session_trial_number.median()
        early = condition_means_from_trials(
            sess, trial_mask=lambda tn, c, m=med: tn <= m)[0]
        late = condition_means_from_trials(
            sess, trial_mask=lambda tn, c, m=med: tn > m)[0]
        halves["first half"].append(early)
        halves["second half"].append(late)
    mats = {k: np.vstack(v) for k, v in halves.items()}
    vlim = np.nanpercentile(np.abs(np.vstack(list(mats.values()))), 98)
    fig, axes = plt.subplots(1, 2, figsize=(9, 7))
    im = None
    for ax, (label, mat) in zip(axes, mats.items()):
        im = _draw(ax, mat, f"{label} of trials (n={mat.shape[0]})", vlim)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="evoked dF/F")
    fig.suptitle("Trial-subset heatmaps rebuilt from single-trial store", fontsize=11)
    fig.savefig(OUTDIR / "heatmap_trial_subset_demo.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "heatmap_trial_subset_demo.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def main():
    plot_combined()
    plot_per_session()
    plot_trial_subset_demo()
    print("Heatmap figures →", OUTDIR)


if __name__ == "__main__":
    main()
