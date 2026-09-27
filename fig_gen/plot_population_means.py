"""Per-condition population-mean figures from tidy CSVs (data/population).

Produces:
  population_means_combined.png    — pooled population mean ± SEM (across sessions), per condition
  population_means_per_session.png — small-multiples grid, one panel per session
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import OUTDIR, load, condition_style, CONDITION_ORDER, annotate_stim

XLIM = (-1.0, 3.0)


def plot_combined():
    df = load("population/population_means_combined.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    annotate_stim(ax)
    for cond in CONDITION_ORDER:
        sub = df[df.condition == cond].sort_values("time_s")
        if sub.empty:
            continue
        st = condition_style(cond)
        ax.plot(sub.time_s, sub.pop_mean_pooled, **st)
        ax.fill_between(sub.time_s,
                        sub.pop_mean_pooled - sub.sem_over_sessions,
                        sub.pop_mean_pooled + sub.sem_over_sessions,
                        color=st["color"], alpha=0.15, lw=0)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Population mean dF/F")
    ax.set_title("Per-condition population mean (pooled neurons, ±SEM across sessions)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "population_means_combined.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "population_means_combined.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_per_session():
    df = load("population/population_means_per_session.csv")
    sessions = list(dict.fromkeys(df.session))
    n = len(sessions)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.4 * nrows),
                             sharex=True, squeeze=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        annotate_stim(ax)
        sdf = df[df.session == sess]
        for cond in CONDITION_ORDER:
            sub = sdf[sdf.condition == cond].sort_values("time_s")
            if sub.empty:
                continue
            st = condition_style(cond)
            ax.plot(sub.time_s, sub.pop_mean, color=st["color"],
                    linestyle=st["linestyle"], lw=1.3)
        ax.set_xlim(*XLIM)
        ax.set_title(sess, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-condition population means by session", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "population_means_per_session.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "population_means_per_session.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_by_projection():
    """Population mean per condition, split by projection status
    (projecting to striatum vs non-projecting)."""
    from pathlib import Path
    from _common import DATA
    path = DATA / "population" / "population_means_by_projection.csv"
    if not Path(path).exists():
        return
    df = load("population/population_means_by_projection.csv")
    groups = [g for g in ["projecting", "nonprojecting"]
              if g in df.projection_status.unique()]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, g in zip(axes, groups):
        annotate_stim(ax)
        gdf = df[df.projection_status == g]
        n = int(gdf.n_neurons_total.iloc[0])
        for cond in CONDITION_ORDER:
            sub = gdf[gdf.condition == cond].sort_values("time_s")
            if sub.empty:
                continue
            st = condition_style(cond)
            ax.plot(sub.time_s, sub.pop_mean_pooled, **st)
            ax.fill_between(sub.time_s,
                            sub.pop_mean_pooled - sub.sem_over_sessions,
                            sub.pop_mean_pooled + sub.sem_over_sessions,
                            color=st["color"], alpha=0.15, lw=0)
        ax.set_xlim(*XLIM)
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.set_title(f"{g} (n={n} neurons)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Population mean dF/F")
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("Per-condition population mean by projection status "
                 "(±SEM across sessions)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "population_means_by_projection.png", dpi=150,
                bbox_inches="tight")
    fig.savefig(str(OUTDIR / "population_means_by_projection.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def main():
    plot_combined()
    plot_per_session()
    plot_by_projection()
    print("Population figures →", OUTDIR)


if __name__ == "__main__":
    main()
