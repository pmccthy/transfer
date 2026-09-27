"""TDR figures from tidy CSVs (results/transfer/data/tdr).

Produces:
  tdr_combined_projections.png  — time-resolved projection onto each TDR axis
  tdr_phase_space.png           — 2D phase-space trajectories (axis pairs)
  tdr_design_matrix.png         — design matrix heatmap
  tdr_variance_explained.png    — per-session PCA-denoising variance explained
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np

from _common import (OUTDIR, load, condition_style, CONDITION_ORDER, smooth,
                     annotate_stim)

PROJ_TYPE = "peak"          # canonical axis choice
SMOOTH_WIN = 3              # samples (~0.2 s at 15 Hz)
XLIM = (-1.0, 3.0)


def _combined():
    df = load("tdr/tdr_combined_projections.csv")
    return df[df.proj_type == PROJ_TYPE]


def plot_time_resolved():
    df = _combined()
    preds = list(dict.fromkeys(df.predictor))
    fig, axes = plt.subplots(len(preds), 1, figsize=(6.5, 2.8 * len(preds)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for pi, (pred, ax) in enumerate(zip(preds, axes)):
        annotate_stim(ax)
        for cond in CONDITION_ORDER:
            sub = df[(df.predictor == pred) & (df.condition == cond)].sort_values("time_s")
            if sub.empty:
                continue
            st = condition_style(cond)
            line_kw = {k: v for k, v in st.items() if k != "label"}
            ax.plot(sub.time_s, smooth(sub.projection.values, SMOOTH_WIN),
                    label=st["label"] if pi == 0 else "_", **line_kw)
        ax.set_ylabel(f"{pred} axis")
        ax.set_xlim(*XLIM)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    fig.suptitle("Combined TDR projections (individual-trial betas, peak axes)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "tdr_combined_projections.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "tdr_combined_projections.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_phase_space():
    df = _combined()
    preds = list(dict.fromkeys(df.predictor))
    pairs = list(itertools.combinations(range(len(preds)), 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 4.6),
                             squeeze=False)
    axes = axes[0]
    for ax, (xi, yi) in zip(axes, pairs):
        for cond in CONDITION_ORDER:
            sx = df[(df.predictor == preds[xi]) & (df.condition == cond)].sort_values("time_s")
            sy = df[(df.predictor == preds[yi]) & (df.condition == cond)].sort_values("time_s")
            if sx.empty:
                continue
            m = (sx.time_s.values >= 0) & (sx.time_s.values <= 3.0)
            st = condition_style(cond)
            x = smooth(sx.projection.values, SMOOTH_WIN)[m]
            y = smooth(sy.projection.values, SMOOTH_WIN)[m]
            ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], lw=st["lw"],
                    label=st["label"])
            ax.scatter(x[0], y[0], color=st["color"], s=25, zorder=3)   # start
            ax.scatter(x[-1], y[-1], color=st["color"], s=45, marker="s",
                       edgecolor="k", lw=0.5, zorder=3)                 # end
        ax.set_xlabel(f"{preds[xi]} axis")
        ax.set_ylabel(f"{preds[yi]} axis")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("TDR phase-space trajectories (0–3 s; ●start ■end)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "tdr_phase_space.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "tdr_phase_space.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_design_matrix():
    df = load("tdr/tdr_design_matrix.csv")
    conds = list(dict.fromkeys(df.condition))
    preds = list(dict.fromkeys(df.predictor))
    M = df.pivot(index="condition", columns="predictor", values="value").loc[conds, preds]
    vmax = max(abs(M.values.min()), abs(M.values.max()), 1e-6)
    fig, ax = plt.subplots(figsize=(max(4, len(preds) * 1.5), max(3, len(conds) * 0.7)))
    im = ax.imshow(M.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="regressor value", fraction=0.035, pad=0.04)
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels(preds, rotation=30, ha="right")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([c.replace(" (pre-Rev)", " pre").replace(" (post-Rev)", " post")
                        for c in conds], fontsize=9)
    for r in range(len(conds)):
        for c in range(len(preds)):
            ax.text(c, r, f"{M.values[r, c]:.2g}", ha="center", va="center",
                    fontsize=8, color="k")
    ax.set_title("TDR design matrix")
    fig.tight_layout()
    fig.savefig(OUTDIR / "tdr_design_matrix.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "tdr_design_matrix.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_variance_explained():
    df = load("tdr/tdr_variance_explained.csv")
    sess = df[df.scope == "session"].copy()
    comb = df[df.scope == "combined"]
    labels = [s.split("_")[-1] + "\n" + s[:10] for s in sess.session]
    fig, ax = plt.subplots(figsize=(max(6, len(sess) * 0.9), 4))
    ax.bar(range(len(sess)), sess.var_explained * 100, color="steelblue",
           edgecolor="k", lw=0.5)
    if not comb.empty:
        ax.axhline(comb.var_explained.iloc[0] * 100, color="crimson", ls="--",
                   label=f"combined ({comb.var_explained.iloc[0]*100:.1f}%)")
        ax.legend(frameon=False)
    ax.set_xticks(range(len(sess)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA denoising variance explained")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "tdr_variance_explained.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "tdr_variance_explained.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def main():
    plot_time_resolved()
    plot_phase_space()
    plot_design_matrix()
    plot_variance_explained()
    print("TDR figures →", OUTDIR)


if __name__ == "__main__":
    main()
