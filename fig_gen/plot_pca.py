"""PCA state-space figures from tidy CSVs (data/pca).

Produces (for the pca_time_avg variant by default):
  pca_pc1_pc2_trajectories.png — PC1-vs-PC2 trajectory per condition (session grid)
  pca_pc_timecourses.png       — PC1..PC3 time courses per condition (combined mean over sessions)
  pca_variance_explained.png   — variance explained per session, both variants
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import OUTDIR, load, condition_style, CONDITION_ORDER, annotate_stim

VARIANT = "pca_time_avg"
XLIM = (-1.0, 3.0)


def plot_pc1_pc2():
    df = load("pca/pca_projections.csv")
    df = df[df.variant == VARIANT]
    sessions = list(dict.fromkeys(df.session))
    n = len(sessions)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows),
                             squeeze=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sdf = df[df.session == sess]
        for cond in CONDITION_ORDER:
            p1 = sdf[(sdf.condition == cond) & (sdf.pc == 1)].sort_values("time_s")
            p2 = sdf[(sdf.condition == cond) & (sdf.pc == 2)].sort_values("time_s")
            if p1.empty:
                continue
            m = (p1.time_s.values >= 0) & (p1.time_s.values <= 3.0)
            st = condition_style(cond)
            x, y = p1.projection.values[m], p2.projection.values[m]
            ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], lw=1.6)
            ax.scatter(x[0], y[0], color=st["color"], s=18, zorder=3)
            ax.scatter(x[-1], y[-1], color=st["color"], s=32, marker="s",
                       edgecolor="k", lw=0.4, zorder=3)
        ax.set_title(sess, fontsize=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"PCA PC1–PC2 trajectories, {VARIANT} (0–3 s; ●start ■end)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pca_pc1_pc2_trajectories.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "pca_pc1_pc2_trajectories.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_pc_timecourses():
    df = load("pca/pca_projections.csv")
    df = df[df.variant == VARIANT]
    # average across sessions for a summary time course
    g = (df[df.pc <= 3]
         .groupby(["condition", "pc", "time_s"], as_index=False)
         .projection.mean())
    pcs = [1, 2, 3]
    fig, axes = plt.subplots(len(pcs), 1, figsize=(6.5, 2.6 * len(pcs)), sharex=True)
    for pc, ax in zip(pcs, axes):
        annotate_stim(ax)
        for cond in CONDITION_ORDER:
            sub = g[(g.pc == pc) & (g.condition == cond)].sort_values("time_s")
            if sub.empty:
                continue
            st = condition_style(cond)
            line_kw = {k: v for k, v in st.items() if k != "label"}
            ax.plot(sub.time_s, sub.projection,
                    label=st["label"] if pc == 1 else "_", **line_kw)
        ax.set_ylabel(f"PC{pc}")
        ax.set_xlim(*XLIM)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    fig.suptitle(f"PCA PC time courses (mean across sessions), {VARIANT}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pca_pc_timecourses.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "pca_pc_timecourses.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_variance():
    df = load("pca/pca_variance_explained.csv")
    variants = list(dict.fromkeys(df.variant))
    sessions = list(dict.fromkeys(df.session))
    x = np.arange(len(sessions))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(max(6, len(sessions) * 0.9), 4))
    for k, variant in enumerate(variants):
        sub = df[df.variant == variant].set_index("session").loc[sessions]
        ax.bar(x + k * w, sub.var_explained * 100, width=w, label=variant,
               edgecolor="k", lw=0.4)
    ax.set_xticks(x + w * (len(variants) - 1) / 2)
    ax.set_xticklabels([s.split("_")[-1] for s in sessions], fontsize=8,
                       rotation=30, ha="right")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA variance explained per session")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pca_variance_explained.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "pca_variance_explained.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def main():
    plot_pc1_pc2()
    plot_pc_timecourses()
    plot_variance()
    print("PCA figures →", OUTDIR)


if __name__ == "__main__":
    main()
