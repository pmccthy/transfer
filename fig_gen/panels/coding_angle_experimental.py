"""coding_angle_experimental.py

Average normalised per-condition coding vectors and their pairwise cosine
similarity ("coding angle"), computed per SESSION from the raw per-neuron
TDR pickles (*_tdr_v2.pkl, see analysis/run_tdr_v2.py), then averaged across
sessions -- the experimental-side analogue of combined/analysis/coding_angle.py
on the RNN side (context-value-RNNs), so the two can be compared directly.

Neuron identity is session-specific (different SNR-filtered cell sets per
session, no cross-session index correspondence -- see run_tdr_v2.py's
load_filtered_cell_indices / the combine_trials() pseudo-population
convention), so cosine similarity is computed WITHIN each session's own
neuron space and only the resulting scalar similarity values are averaged
across sessions -- never the raw coding vectors themselves.

Uses all sessions with a *_tdr_v2.pkl present (13/13 currently), matching the
established "no decode-accuracy filter" convention from behaviour.py's
draw_reversal_lick_combined (see that file's docstring for why the filter
was reverted).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _tags import new_panel, save_panel

try:
    from sat_plot_colours import STIM_STEM_COLOURS
except Exception:
    STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}

DEFAULT_RESULTS_DIR = "~/Documents/experimental_results/08_04_26_dff_from_flu_processed_zscore_snr_tdr_v2"
EXPERIMENTAL_RESULTS_DIR = Path(
    os.environ.get("EXPERIMENTAL_RESULTS_DIR", DEFAULT_RESULTS_DIR)
).expanduser()

STEM_ORDER = ["100_to_0", "50", "0_to_100"]
STEM_DISPLAY = {"100_to_0": "100%->0%", "50": "50%", "0_to_100": "0%->100%"}
# trial_type_list is always [100->0 pre, 50 pre, 0->100 pre, 100->0 post, 50 post, 0->100 post]
PRE_IDX = [0, 1, 2]
POST_IDX = [3, 4, 5]

_CACHE: dict = {}


def _session_files():
    return sorted(EXPERIMENTAL_RESULTS_DIR.glob("*_tdr_v2.pkl"))


def _condition_coding_vectors(d: dict) -> np.ndarray:
    """d: one loaded *_tdr_v2.pkl. Returns (6, n_neurons) L2-normalised coding vectors,
    one row per entry of d['trial_type_list'], averaged over the stim-presentation window."""
    fs = d["fs_image"]
    onset = d["stim_onset_abs"]
    win = slice(onset, onset + int(round(d["stim_duration_s"] * fs)))
    vecs = d["cond_avg"][:, :, win].mean(axis=2)  # (6, n_neurons)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return vecs / norms


def _all_session_matrices():
    if "mats" not in _CACHE:
        mats, names = [], []
        for f in _session_files():
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            vecs = _condition_coding_vectors(d)
            mats.append(vecs @ vecs.T)  # (6,6)
            names.append(d["session_name"])
        if not mats:
            raise FileNotFoundError(f"no *_tdr_v2.pkl found under {EXPERIMENTAL_RESULTS_DIR}")
        _CACHE["mats"] = np.asarray(mats)
        _CACHE["names"] = names
    return _CACHE["mats"], _CACHE["names"]


def coding_angle_summary_experimental():
    """Mean +/- SEM (6,6) cosine-similarity matrix across sessions."""
    mats, names = _all_session_matrices()
    mean = mats.mean(axis=0)
    sem = mats.std(axis=0, ddof=1) / np.sqrt(mats.shape[0])
    return mean, sem, mats.shape[0]


def draw_coding_angle_heatmap(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.2, 5.2))
    mean, sem, n_sessions = coding_angle_summary_experimental()
    labels = ["100->0% (pre)", "50% (pre)", "0->100% (pre)", "100->0% (post)", "50% (post)", "0->100% (post)"]
    im = ax.imshow(mean, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(mean[i, j]) > 0.5 else "black")
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.set_title(f"Experimental coding-vector angle  [n={n_sessions} sessions]")
    return fig


def draw_coding_angle_heatmap_triangular(ax=None, colorbar=None):
    """Same combined 6-condition (3 stem-pairs x pre/post) heatmap as
    draw_coding_angle_heatmap(), but triangular -- the redundant upper half
    masked out, matching the convention used on the RNN side (see
    context-value-RNNs' combined/panels/coding_angle_panel.py's
    draw_coding_angle_heatmap_combined) so the two are visually comparable
    apples-to-apples. The matrix is exactly symmetric with an exactly-1.0
    diagonal by construction (vecs @ vecs.T on unit-normalised rows)."""
    fig, ax, owns = new_panel(ax, figsize=(5.2, 5.2))
    if colorbar is None:
        colorbar = owns
    mean, sem, n_sessions = coding_angle_summary_experimental()
    labels = ["100->0% (pre)", "50% (pre)", "0->100% (pre)", "100->0% (post)", "50% (post)", "0->100% (post)"]
    n = len(labels)
    hide = np.triu(np.ones((n, n), dtype=bool), k=1)
    cmap = plt.get_cmap("Reds").copy()  # sequential white->red -- values are all
    # positive in practice, so no need for a diverging colormap's blue half
    cmap.set_bad(color="white")
    masked = np.ma.masked_where(hide, mean)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap)  # all-positive similarities in
    # practice -- crop to [0, 1] instead of [-1, 1] so the full colormap range is used
    for i in range(n):
        for j in range(n):
            if hide[i, j]:
                continue
            txt = f"{mean[i, j]:.2f}" if i != j else "1.00"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="white" if abs(mean[i, j]) > 0.5 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel("cosine similarity")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    # title deliberately omitted -- matches the RNN-side combined heatmap convention
    # (compose.py's _clear_heatmap_titles), the y-label already says what this is.
    return fig


def draw_coding_angle_by_phase(ax=None):
    """Bar chart: for each stem-pair, one bar for pre-Rev and one for post-Rev, mean +/- SEM across sessions."""
    fig, ax, _ = new_panel(ax, figsize=(5.6, 4.5))
    mats, names = _all_session_matrices()
    pairs = [(0, 1), (0, 2), (1, 2)]  # indices into STEM_ORDER
    pair_labels = [f"{STEM_DISPLAY[STEM_ORDER[i]]} vs {STEM_DISPLAY[STEM_ORDER[j]]}" for i, j in pairs]

    x = np.arange(len(pairs))
    width = 0.35
    for pi, (idx_set, label, color) in enumerate([(PRE_IDX, "pre-Rev", "0.35"), (POST_IDX, "post-Rev", "#c24167")]):
        vals, errs = [], []
        for i, j in pairs:
            v = mats[:, idx_set[i], idx_set[j]]
            vals.append(v.mean())
            errs.append(v.std(ddof=1) / np.sqrt(len(v)))
        ax.bar(x + (pi - 0.5) * width, vals, width, yerr=errs, color=color, label=label, capsize=3)

    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cosine similarity")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Coding-vector angle: pre vs post-reversal  [n={mats.shape[0]} sessions]")
    return fig


def build_all(show_tag=None):
    try:
        fig = draw_coding_angle_heatmap()
    except FileNotFoundError as e:
        print(f"  (skip experimental coding-angle heatmap: {e})")
    else:
        save_panel(fig, "Dimensionality/CodingAngle", "DIM.ANGLE.experimental",
                   "coding_angle_heatmap_experimental", show_tag)

    try:
        fig = draw_coding_angle_heatmap_triangular()
    except FileNotFoundError as e:
        print(f"  (skip experimental coding-angle heatmap [triangular]: {e})")
    else:
        save_panel(fig, "Dimensionality/CodingAngle", "DIM.ANGLE.experimental.triangular",
                   "coding_angle_heatmap_experimental_triangular", show_tag)

    try:
        fig = draw_coding_angle_by_phase()
    except FileNotFoundError as e:
        print(f"  (skip experimental coding-angle by-phase bars: {e})")
    else:
        save_panel(fig, "Dimensionality/CodingAngle", "DIM.ANGLE.experimental.byphase",
                   "coding_angle_by_phase_experimental", show_tag)


if __name__ == "__main__":
    build_all()
