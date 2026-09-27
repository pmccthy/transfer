"""Dimensionality panels (from data/dimensionality/*.csv and data/tdr/*).

Tags:
  DIM.EXP.eigen     Expert eigenspectrum (explained-variance spectrum), all sessions
  DIM.EXP.npc90     Expert PCs-for-90% variance, bar per session
  DIM.EXP.pr        Expert participation ratio, bar per session
  DIM.EXP.PCA.proj  Expert PC-pair projections (per session -> one file each)
  DIM.REV.eigen     Reversal eigenspectrum, pre & post
  DIM.REV.npc90     Reversal PCs-for-90%, pre/post bars per session
  DIM.REV.pr        Reversal participation ratio, pre/post bars per session
  DIM.REV.PCA.proj  Reversal PC-pair projections (per session, pre & post)
  DIM.REV.TDR.proj  Reversal TDR-component-pair projections (per session)
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _tags import DATA, new_panel, save_panel
from _common import smooth as _smooth

try:
    from sat_plot_colours import STIM_STEM_COLOURS
except Exception:
    STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}

DIM = DATA / "dimensionality"
SPEC = DIM / "dimensionality_eigenspectrum.csv"
SUMM = DIM / "dimensionality_summary.csv"
PROJ = DIM / "dimensionality_pca_projections.csv"
TDR_PS = DATA / "tdr" / "tdr_per_session_projections.csv"

EXP_COL = {"100": STIM_STEM_COLOURS["100_to_0"], "50": STIM_STEM_COLOURS["50"], "0": STIM_STEM_COLOURS["0_to_100"]}
REV_COL = {"100_to_0": STIM_STEM_COLOURS["100_to_0"], "50": STIM_STEM_COLOURS["50"], "0_to_100": STIM_STEM_COLOURS["0_to_100"]}
PHASE_COL = {"pre": "#1f77b4", "post": "#d62728"}


def _short(s):
    return s.split("_")[-1] + " " + s[:10]


# ---- eigenspectrum ---------------------------------------------------------
def draw_eigenspectrum(scope, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6, 4.5))
    df = pd.read_csv(SPEC).query("scope == @scope")
    phases = ["pre", "post"] if scope == "reversal" else ["expert"]
    for phase in phases:
        pdf = df[df.phase == phase]
        col = PHASE_COL.get(phase, "0.3")
        for sess, g in pdf.groupby("session"):
            g = g.sort_values("pc_index")
            ax.plot(g.pc_index, g.evr, color=col, lw=0.6, alpha=0.3)
        mean = pdf.groupby("pc_index").evr.mean()
        ax.plot(mean.index, mean.values, color=col, lw=2.2,
                label=phase if scope == "reversal" else "mean")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)   # ignore rank-deficient numerical floor at the tail
    ax.set_xlabel("PC index"); ax.set_ylabel("explained variance ratio")
    ax.set_title(f"{scope.capitalize()} eigenspectrum (per session + mean)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---- summary bars (nPC90, participation ratio) -----------------------------
def _summary_bar(scope, metric, ylabel, title, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(max(6, 0.0), 4.5))
    df = pd.read_csv(SUMM).query("scope == @scope")
    sessions = sorted(df.session.unique())
    fig.set_size_inches(max(6, len(sessions) * 0.5), 4.5)
    if scope == "reversal":
        w = 0.4
        for k, phase in enumerate(["pre", "post"]):
            vals = [df[(df.session == s) & (df.phase == phase)][metric].values for s in sessions]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(np.arange(len(sessions)) + (-w/2 if phase == "pre" else w/2), vals,
                   width=w, color=PHASE_COL[phase], label=phase)
        ax.legend(frameon=False, fontsize=8)
    else:
        vals = [df[df.session == s][metric].values[0] for s in sessions]
        ax.bar(range(len(sessions)), vals, color="steelblue")
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([s.split("_")[-1] for s in sessions], rotation=90, fontsize=6)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_npc90(scope, ax=None):
    return _summary_bar(scope, "n_pcs_90", "PCs for 90% variance",
                        f"{scope.capitalize()}: dimensionality (PCs for 90%)", ax)


def draw_participation(scope, ax=None):
    return _summary_bar(scope, "participation_ratio", "participation ratio",
                        f"{scope.capitalize()}: participation ratio", ax)


# ---- per-session PCA projections -------------------------------------------
def draw_pca_projections(scope, session, smooth_window=5):
    df = pd.read_csv(PROJ).query("scope == @scope and session == @session")
    phases = ["pre", "post"] if scope == "reversal" else ["expert"]
    col = REV_COL if scope == "reversal" else EXP_COL
    pairs = [(1, 2), (1, 3), (2, 3)]
    fig, axes = plt.subplots(len(phases), 3, figsize=(4.2 * 3, 4.0 * len(phases)),
                             squeeze=False)
    for r, phase in enumerate(phases):
        pdf = df[df.phase == phase]
        for c, (xi, yi) in enumerate(pairs):
            ax = axes[r][c]
            for stim in col:
                sx = pdf[(pdf.stimulus.astype(str) == stim) & (pdf.pc == xi)].sort_values("time_s")
                sy = pdf[(pdf.stimulus.astype(str) == stim) & (pdf.pc == yi)].sort_values("time_s")
                if sx.empty:
                    continue
                m = (sx.time_s.values >= 0) & (sx.time_s.values <= 3.0)
                x, y = sx.projection.values[m], sy.projection.values[m]
                x, y = _smooth(x, smooth_window), _smooth(y, smooth_window)
                ax.plot(x, y, color=col[stim], lw=1.6)
                ax.scatter(x[0], y[0], color=col[stim], s=18, zorder=3)
                ax.scatter(x[-1], y[-1], color=col[stim], s=32, marker="s",
                           edgecolor="k", lw=0.4, zorder=3)
            ax.set_xlabel(f"PC{xi}"); ax.set_ylabel(f"PC{yi}")
            ax.set_title(f"{phase} PC{xi}-PC{yi}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{scope} PCA projections — {session} (0–3 s; ●start ■end)", fontsize=10)
    fig.tight_layout()
    return fig


# ---- per-session TDR projections (reversal) --------------------------------
def draw_tdr_projections(session, smooth_window=5):
    df = pd.read_csv(TDR_PS).query("session == @session")
    preds = list(dict.fromkeys(df.predictor))
    pairs = list(itertools.combinations(range(len(preds)), 2))
    from _common import condition_style, CONDITION_ORDER
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.4), squeeze=False)
    axes = axes[0]
    for ax, (xi, yi) in zip(axes, pairs):
        for cond in CONDITION_ORDER:
            sx = df[(df.predictor == preds[xi]) & (df.condition == cond)].sort_values("time_s")
            sy = df[(df.predictor == preds[yi]) & (df.condition == cond)].sort_values("time_s")
            if sx.empty:
                continue
            m = (sx.time_s.values >= 0) & (sx.time_s.values <= 3.0)
            st = condition_style(cond)
            x, y = sx.projection.values[m], sy.projection.values[m]
            x, y = _smooth(x, smooth_window), _smooth(y, smooth_window)
            ax.plot(x, y, color=st["color"], ls=st["linestyle"], lw=1.6)
            ax.scatter(x[-1], y[-1], color=st["color"], s=28, marker="s", edgecolor="k", lw=0.4, zorder=3)
        ax.set_xlabel(f"{preds[xi]} axis"); ax.set_ylabel(f"{preds[yi]} axis")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"TDR component projections — {session} (0–3 s)", fontsize=10)
    fig.tight_layout()
    return fig


def _representative_subset(sessions, n=12):
    """Evenly-spaced subset of `n` sessions across the sorted full list --
    used for the "expert" PCA grid below (too many sessions -- 43 -- for one
    exhaustive grid, per the chat; "reversal" (13 sessions) always gets the
    full set)."""
    sessions = sorted(sessions)
    if len(sessions) <= n:
        return sessions
    idx = sorted(set(np.linspace(0, len(sessions) - 1, n).round().astype(int)))
    return [sessions[i] for i in idx]


def draw_pca_projections_grid(scope, sessions=None, pc_pair=(1, 2), ncols=6, smooth_window=5):
    """Supplementary grid (record of every session, per the chat): one panel
    per session, single PC pair (default PC1-PC2 -- the full 3-pair detail
    per session is already in draw_pca_projections()'s own per-session
    files, see build_all() below), phase-space trajectory over 0-3s. Pre
    solid / post dashed when scope=="reversal" (no phase split for
    "expert" -- one line). `sessions` defaults to every session in this
    scope; pass a subset (see _representative_subset) for "expert", which
    has too many (43) for one exhaustive grid."""
    df = pd.read_csv(PROJ).query("scope == @scope")
    all_sessions = sorted(df.session.unique())
    sessions = sessions if sessions is not None else all_sessions
    phases = ["pre", "post"] if scope == "reversal" else ["expert"]
    col = REV_COL if scope == "reversal" else EXP_COL
    xi, yi = pc_pair
    n = len(sessions)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sdf = df[df.session == sess]
        for phase in phases:
            pdf = sdf[sdf.phase == phase]
            ls = "--" if phase == "post" else "-"
            for stim in col:
                sx = pdf[(pdf.stimulus.astype(str) == stim) & (pdf.pc == xi)].sort_values("time_s")
                sy = pdf[(pdf.stimulus.astype(str) == stim) & (pdf.pc == yi)].sort_values("time_s")
                if sx.empty:
                    continue
                m = (sx.time_s.values >= 0) & (sx.time_s.values <= 3.0)
                x, y = sx.projection.values[m], sy.projection.values[m]
                if len(x) == 0:
                    continue
                x, y = _smooth(x, smooth_window), _smooth(y, smooth_window)
                ax.plot(x, y, color=col[stim], ls=ls, lw=1.2)
                ax.scatter(x[-1], y[-1], color=col[stim], s=18, marker="s",
                           edgecolor="k", lw=0.3, zorder=3)
        ax.set_title(sess.split("_")[-1] + " " + sess[:10], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    cov = "every session" if len(sessions) == len(all_sessions) else f"{len(sessions)} of {len(all_sessions)} sessions"
    fig.suptitle(f"{scope.capitalize()} PCA phase space (PC{xi}-PC{yi}, 0-3s) -- {cov}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def draw_tdr_projections_grid(sessions=None, pred_pair=(0, 1), ncols=6, smooth_window=5):
    """Supplementary grid (record of every session, per the chat): grid
    version of draw_tdr_projections() -- one panel per reversal session (all
    13 -- TDR data is reversal-only, so no expert-side subset issue here),
    single predictor pair (default the first two, e.g. stim vs context)."""
    from _common import condition_style, CONDITION_ORDER
    df = pd.read_csv(TDR_PS)
    all_sessions = sorted(df.session.unique())
    sessions = sessions if sessions is not None else all_sessions
    preds = list(dict.fromkeys(df.predictor))
    xi, yi = pred_pair
    n = len(sessions)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sdf = df[df.session == sess]
        for cond in CONDITION_ORDER:
            sx = sdf[(sdf.predictor == preds[xi]) & (sdf.condition == cond)].sort_values("time_s")
            sy = sdf[(sdf.predictor == preds[yi]) & (sdf.condition == cond)].sort_values("time_s")
            if sx.empty:
                continue
            m = (sx.time_s.values >= 0) & (sx.time_s.values <= 3.0)
            st = condition_style(cond)
            x, y = sx.projection.values[m], sy.projection.values[m]
            if len(x) == 0:
                continue
            x, y = _smooth(x, smooth_window), _smooth(y, smooth_window)
            ax.plot(x, y, color=st["color"], ls=st["linestyle"], lw=1.2)
            ax.scatter(x[-1], y[-1], color=st["color"], s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
        ax.set_title(sess.split("_")[-1] + " " + sess[:10], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Reversal TDR phase space ({preds[xi]} vs {preds[yi]}, 0-3s) -- every session",
                fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def build_all(show_tag=None):
    if not SUMM.exists():
        print("  (no dimensionality data — run extract/extract_dimensionality.py)")
        return
    for scope, folder in [("expert", "Dimensionality/Expert"), ("reversal", "Dimensionality/Reversal")]:
        tag = "EXP" if scope == "expert" else "REV"
        save_panel(draw_eigenspectrum(scope), folder, f"DIM.{tag}.eigen", f"{scope}_eigenspectrum", show_tag)
        save_panel(draw_npc90(scope), folder, f"DIM.{tag}.npc90", f"{scope}_npc90", show_tag)
        save_panel(draw_participation(scope), folder, f"DIM.{tag}.pr", f"{scope}_participation_ratio", show_tag)
    # per-session PCA projections
    proj = pd.read_csv(PROJ)
    for scope, folder in [("expert", "Dimensionality/Expert/PCA"), ("reversal", "Dimensionality/Reversal/PCA")]:
        tag = "EXP" if scope == "expert" else "REV"
        for sess in sorted(proj[proj.scope == scope].session.unique()):
            save_panel(draw_pca_projections(scope, sess), folder, f"DIM.{tag}.PCA.proj",
                       f"{scope}_pca_proj_{sess}", show_tag)
    # per-session TDR projections (reversal)
    tdr = pd.read_csv(TDR_PS)
    for sess in sorted(tdr.session.unique()):
        save_panel(draw_tdr_projections(sess), "Dimensionality/Reversal/TDR",
                   "DIM.REV.TDR.proj", f"reversal_tdr_proj_{sess}", show_tag)
    # supplementary multi-session grids -- record of every session (reversal: all
    # 13; expert: a representative 12-of-43 subset, too many for one grid)
    save_panel(draw_pca_projections_grid("reversal"), "Dimensionality/Reversal/PCA",
               "DIM.REV.PCA.grid", "reversal_pca_grid_all_sessions", show_tag)
    expert_sessions = sorted(proj[proj.scope == "expert"].session.unique())
    save_panel(draw_pca_projections_grid("expert", sessions=_representative_subset(expert_sessions)),
               "Dimensionality/Expert/PCA", "DIM.EXP.PCA.grid", "expert_pca_grid_subset", show_tag)
    save_panel(draw_tdr_projections_grid(), "Dimensionality/Reversal/TDR",
               "DIM.REV.TDR.grid", "reversal_tdr_grid_all_sessions", show_tag)


if __name__ == "__main__":
    build_all()
