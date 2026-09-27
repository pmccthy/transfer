"""Population-activity panels (time-resolved mean dF/F per stimulus).

Tags:
  POP.REV.mean      Reversal: 3 stimuli, pre & post overlaid (± SEM across sessions)
  POP.REV.meanproj  Reversal: same, split by projection status (projecting/non-)
  POP.EXP.mean      Expert: 3 stimuli (± SEM across sessions)      [needs expert data]
  POP.EXP.meanproj  Expert: same, split by projection status       [needs expert data]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _tags import DATA, new_panel, save_panel
from _common import condition_style, CONDITION_ORDER, annotate_stim  # noqa: E402

XLIM = (-1.0, 3.0)

REV_COMB = DATA / "population" / "population_means_combined.csv"
REV_PROJ = DATA / "population" / "population_means_by_projection.csv"
EXP_COMB = DATA / "population" / "population_means_expert.csv"
EXP_PROJ = DATA / "population" / "population_means_expert_by_projection.csv"

EXP_ORDER = ["100", "50", "0"]
EXP_PRETTY = {"100": "100%", "50": "50%", "0": "0%"}


def _style_expert(lab):
    try:
        from sat_plot_colours import STIM_STEM_COLOURS
    except Exception:
        STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}
    m = {"100": STIM_STEM_COLOURS["100_to_0"], "50": STIM_STEM_COLOURS["50"],
         "0": STIM_STEM_COLOURS["0_to_100"]}
    return dict(color=m[lab], lw=2.0, label=EXP_PRETTY[lab])


# ---- reversal (POP.REV.mean) -----------------------------------------------
def draw_reversal_mean(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(REV_COMB)
    annotate_stim(ax)
    for cond in CONDITION_ORDER:
        sub = df[df.condition == cond].sort_values("time_s")
        if sub.empty:
            continue
        st = condition_style(cond)
        ax.plot(sub.time_s, sub.pop_mean_pooled, **st)
        ax.fill_between(sub.time_s, sub.pop_mean_pooled - sub.sem_over_sessions,
                        sub.pop_mean_pooled + sub.sem_over_sessions,
                        color=st["color"], alpha=0.15, lw=0)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Population activity ($\Delta$F/F)")
    ax.set_title("Reversal population mean (3 stimuli × pre/post, ± SEM)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    return fig


def draw_reversal_mean_projection(ax=None):
    df = pd.read_csv(REV_PROJ)
    groups = [g for g in ["projecting", "nonprojecting"] if g in df.projection_status.unique()]
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
            ax.fill_between(sub.time_s, sub.pop_mean_pooled - sub.sem_over_sessions,
                            sub.pop_mean_pooled + sub.sem_over_sessions,
                            color=st["color"], alpha=0.15, lw=0)
        ax.set_xlim(*XLIM); ax.set_xlabel("Time from stimulus onset (s)")
        ax.set_title(f"{g} (n={n})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Population activity ($\Delta$F/F)")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Reversal population mean by projection status", fontsize=11)
    fig.tight_layout()
    return fig


# ---- expert (POP.EXP.mean / meanproj) --------------------------------------
def draw_expert_mean(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(EXP_COMB)
    annotate_stim(ax)
    for lab in EXP_ORDER:
        sub = df[df.stimulus.astype(str) == lab].sort_values("time_s")
        if sub.empty:
            continue
        st = _style_expert(lab)
        ax.plot(sub.time_s, sub.pop_mean_pooled, **st)
        ax.fill_between(sub.time_s, sub.pop_mean_pooled - sub.sem_over_sessions,
                        sub.pop_mean_pooled + sub.sem_over_sessions,
                        color=st["color"], alpha=0.15, lw=0)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Population activity ($\Delta$F/F)")
    ax.set_title("Expert population mean (3 stimuli, ± SEM)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    return fig


def draw_expert_mean_projection(ax=None):
    df = pd.read_csv(EXP_PROJ)
    groups = [g for g in ["projecting", "nonprojecting"] if g in df.projection_status.unique()]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, g in zip(axes, groups):
        annotate_stim(ax)
        gdf = df[df.projection_status == g]
        n = int(gdf.n_neurons_total.iloc[0])
        for lab in EXP_ORDER:
            sub = gdf[gdf.stimulus.astype(str) == lab].sort_values("time_s")
            if sub.empty:
                continue
            st = _style_expert(lab)
            ax.plot(sub.time_s, sub.pop_mean_pooled, **st)
            ax.fill_between(sub.time_s, sub.pop_mean_pooled - sub.sem_over_sessions,
                            sub.pop_mean_pooled + sub.sem_over_sessions,
                            color=st["color"], alpha=0.15, lw=0)
        ax.set_xlim(*XLIM); ax.set_xlabel("Time from stimulus onset (s)")
        ax.set_title(f"{g} (n={n})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Population activity ($\Delta$F/F)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Expert population mean by projection status", fontsize=11)
    fig.tight_layout()
    return fig


def build_all(show_tag=None):
    if REV_COMB.exists():
        save_panel(draw_reversal_mean(), "Population activity/Reversal", "POP.REV.mean",
                   "reversal_pop_mean", show_tag)
    if REV_PROJ.exists():
        save_panel(draw_reversal_mean_projection(), "Population activity/Reversal",
                   "POP.REV.meanproj", "reversal_pop_mean_projection", show_tag)
    if EXP_COMB.exists():
        save_panel(draw_expert_mean(), "Population activity/Expert", "POP.EXP.mean",
                   "expert_pop_mean", show_tag)
    if EXP_PROJ.exists():
        save_panel(draw_expert_mean_projection(), "Population activity/Expert",
                   "POP.EXP.meanproj", "expert_pop_mean_projection", show_tag)


if __name__ == "__main__":
    build_all()
