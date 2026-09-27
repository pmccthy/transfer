"""Time-pooled decoding panels (from data/decoding/decoding_timepooled*.csv).

Tags:
  DEC.EXP.TP.stimpair    Expert stimulus-pair accuracy bar
  DEC.REV.TP.stimpair    Reversal stimulus-pair accuracy bar (pre & post)
  DEC.REV.TP.stimscatter Reversal stim pairs: pre vs post, lines joining, SEM bars
  DEC.REV.TP.transferPP  Pre->post transfer, scatter style (within-pre vs pre->post test)
  DEC.REV.TP.transferPr  Post->pre transfer, scatter style (within-post vs post->pre test)
  DEC.REV.TP.context     Context (pre vs post) bar
  DEC.REV.TP.value       Value decoding bar (value_xor)
  DEC.REV.TP.stimidentity Stimulus-identity decoding bar (stim_identity)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _tags import DATA, new_panel, save_panel

TP = DATA / "decoding" / "decoding_timepooled.csv"
TP_PS = DATA / "decoding" / "decoding_timepooled_per_session.csv"
TP_CROSS = DATA / "decoding" / "decoding_timepooled_cross.csv"
CHANCE = 0.5

PAIR_COL = {"0v1": "#fb8072", "0v50": "#bebada", "1v50": "#8dd3c7"}
PAIR_LAB = {"0v1": "0% vs 100%", "0v50": "0% vs 50%", "1v50": "100% vs 50%"}
# stacked tick labels (X \n vs \n X) so tick fonts can be larger
PAIR_XLAB = {"0v1": "0%\nvs\n100%", "0v50": "0%\nvs\n50%", "1v50": "100%\nvs\n50%"}
PAIRS = ["0v1", "1v50", "0v50"]


def _within(scope):
    return pd.read_csv(TP).query("scope == @scope").set_index("decoder")


def _persess(scope):
    return pd.read_csv(TP_PS).query("scope == @scope")


def _chance(ax):
    ax.axhline(CHANCE, color="0.5", ls="--", lw=1.8, zorder=1, label="chance")


def _points(ax, x, scope, decoder, color):
    ps = _persess(scope)
    vals = ps[ps.decoder == decoder].accuracy.values
    if len(vals):
        j = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.22
        ax.scatter(x + j, vals, color="0.3", s=11, alpha=0.5, zorder=3)


# ---- expert stim-pair bar --------------------------------------------------
def draw_expert_stimpair_bar(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    w = _within("expert")
    _chance(ax)
    for x, p in enumerate(PAIRS):
        d = f"stim_{p}"
        ax.bar(x, w.loc[d, "mean_accuracy"], width=0.65, color=PAIR_COL[p], yerr=w.loc[d, "sem"], capsize=4, zorder=2)
        _points(ax, x, "expert", d, PAIR_COL[p])
    ax.set_xticks(range(len(PAIRS))); ax.set_xticklabels([PAIR_XLAB[p] for p in PAIRS])
    ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Expert stimulus-pair decoding (time-pooled)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---- reversal stim-pair bar (pre & post) -----------------------------------
def draw_reversal_stimpair_bar(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    w = _within("reversal")
    _chance(ax)
    width = 0.38
    for i, p in enumerate(PAIRS):
        for k, (phase, off) in enumerate([("pre", -width / 2), ("post", width / 2)]):
            d = f"stim_{phase}_{p}"
            ax.bar(i + off, w.loc[d, "mean_accuracy"], width=width,
                   color=PAIR_COL[p], alpha=1.0 if phase == "pre" else 0.55,
                   yerr=w.loc[d, "sem"], capsize=3, zorder=2,
                   label=("pre" if i == 0 and phase == "pre" else
                          "post" if i == 0 and phase == "post" else None))
    ax.set_xticks(range(len(PAIRS))); ax.set_xticklabels([PAIR_XLAB[p] for p in PAIRS])
    ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Reversal stimulus-pair decoding (time-pooled; pre solid, post faded)", fontsize=9)
    # This panel keeps its own local pre/post legend rather than folding into the
    # composite's shared legend (see compose.py's figure2() exempt_legend_axes) --
    # bigger font per request, and drop the "chance" entry (the dashed line itself
    # still renders via _chance(ax) above, just not named in this legend).
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l != "chance"]
    ax.legend(*zip(*keep), frameon=False, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---- reversal stim scatter (pre vs post, lines joining) --------------------
def draw_reversal_stim_scatter(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    w = _within("reversal")
    _chance(ax)
    for p in PAIRS:
        pre, post = f"stim_pre_{p}", f"stim_post_{p}"
        y = [w.loc[pre, "mean_accuracy"], w.loc[post, "mean_accuracy"]]
        e = [w.loc[pre, "sem"], w.loc[post, "sem"]]
        ax.errorbar([0, 1], y, yerr=e, color=PAIR_COL[p], marker="o", lw=2,
                    capsize=4, label=PAIR_LAB[p])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pre", "post"])
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Reversal stimulus-pair: pre vs post", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---- transfer scatter (within vs cross-test) -------------------------------
def _draw_transfer_scatter(direction, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    w = _within("reversal")
    cross = pd.read_csv(TP_CROSS).query("split == 'test'").set_index("decoder")
    _chance(ax)
    train_phase = "pre" if direction == "pre2post" else "post"
    for p in PAIRS:
        within = f"stim_{train_phase}_{p}"
        cd = f"stim_cross_{p}_{direction}"
        y = [w.loc[within, "mean_accuracy"], cross.loc[cd, "mean_accuracy"]]
        e = [w.loc[within, "sem"], cross.loc[cd, "sem"]]
        ax.errorbar([0, 1], y, yerr=e, color=PAIR_COL[p], marker="o", lw=2,
                    capsize=4, label=PAIR_LAB[p])
    xt = [f"train {train_phase}", "test " + ("post" if direction == "pre2post" else "pre")]
    ax.set_xticks([0, 1]); ax.set_xticklabels(xt)
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reversal transfer {train_phase}→"
                 f"{'post' if direction=='pre2post' else 'pre'} (time-pooled)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_reversal_transfer_pp(ax=None):
    return _draw_transfer_scatter("pre2post", ax)


def draw_reversal_transfer_pr(ax=None):
    return _draw_transfer_scatter("post2pre", ax)


# ---- context / value / stim-identity single-ish bars -----------------------
def draw_reversal_context_bar(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    w = _within("reversal")
    _chance(ax)
    decs = [("context_100to0", "#4b2362", "100%→0%"),
            ("phase_50_pre_vs_post", "#c24167", "50%"),
            ("context_0to100", "#edb081", "0%→100%")]
    for x, (d, c, lab) in enumerate(decs):
        ax.bar(x, w.loc[d, "mean_accuracy"], width=0.65, color=c, yerr=w.loc[d, "sem"], capsize=4, zorder=2)
        _points(ax, x, "reversal", d, c)
    ax.set_xticks(range(len(decs))); ax.set_xticklabels([d[2] for d in decs])
    ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Reversal context (pre vs post) decoding", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _single_bar(decoder, title, color, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(3.2, 4.5))
    w = _within("reversal")
    _chance(ax)
    ax.bar(0, w.loc[decoder, "mean_accuracy"], width=0.6, color=color, yerr=w.loc[decoder, "sem"], capsize=4, zorder=2)
    _points(ax, 0, "reversal", decoder, color)
    ax.set_xticks([0]); ax.set_xticklabels([title.split("\n")[0]])
    ax.set_xlim(-0.7, 0.7); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("accuracy"); ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_reversal_value_bar(ax=None):
    return _single_bar("value_xor", "Value\n(100% vs 0%, pooled)", "#1f77b4", ax)


def draw_reversal_stimidentity_bar(ax=None):
    return _single_bar("stim_identity", "Stimulus identity\n(100→0 vs 0→100 cue)", "#d62728", ax)


def build_all(show_tag=None):
    if not TP.exists():
        print("  (no time-pooled decoding data — skipping DEC.*.TP.*)")
        return
    et = "Decoding/Expert/Time-pooled"
    rt = "Decoding/Reversal/Time-pooled"
    save_panel(draw_expert_stimpair_bar(), et, "DEC.EXP.TP.stimpair", "expert_stimpair_bar", show_tag)
    save_panel(draw_reversal_stimpair_bar(), rt, "DEC.REV.TP.stimpair", "reversal_stimpair_bar", show_tag)
    save_panel(draw_reversal_stim_scatter(), rt, "DEC.REV.TP.stimscatter", "reversal_stim_scatter", show_tag)
    save_panel(draw_reversal_transfer_pp(), rt, "DEC.REV.TP.transferPP", "reversal_transfer_pre2post", show_tag)
    save_panel(draw_reversal_transfer_pr(), rt, "DEC.REV.TP.transferPr", "reversal_transfer_post2pre", show_tag)
    save_panel(draw_reversal_context_bar(), rt, "DEC.REV.TP.context", "reversal_context_bar", show_tag)
    save_panel(draw_reversal_value_bar(), rt, "DEC.REV.TP.value", "reversal_value_bar", show_tag)
    save_panel(draw_reversal_stimidentity_bar(), rt, "DEC.REV.TP.stimidentity", "reversal_stimidentity_bar", show_tag)


if __name__ == "__main__":
    build_all()
