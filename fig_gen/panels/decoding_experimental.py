"""Decoding panels.

Time-resolved (from data/decoding/decoding_*.csv):
  DEC.EXP.TR.stimpair   Expert stimulus-pair accuracy
  DEC.REV.TR.stimpair   Reversal stimulus-pair accuracy (pre & post)
  DEC.REV.TR.transferPP Cross-phase pre->post generalisation
  DEC.REV.TR.transferPr Cross-phase post->pre generalisation
  DEC.REV.TR.context    Context (pre vs post) decoding
  DEC.REV.TR.value      Value (value_xor) decoding

Time-pooled panels are built in panels/decoding_pooled.py (needs the canonical
time-pooled decoder output).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _tags import DATA, new_panel, save_panel
from _common import annotate_stim  # noqa: E402

CHANCE = 0.5
XLIM = (-1.0, 3.0)
AGG = DATA / "decoding" / "decoding_aggregated.csv"
CROSS = DATA / "decoding" / "decoding_cross_phase.csv"
EXPERT = DATA / "decoding" / "decoding_expert.csv"

# stimulus-pair colours (match sat_plot_colours DECODER_PREFIX_LINE_COLOURS)
PAIR_COL = {"0v1": "#fb8072", "0v50": "#bebada", "1v50": "#8dd3c7"}
PAIR_LAB = {"0v1": "0% vs 100%", "0v50": "0% vs 50%", "1v50": "100% vs 50%"}


def _band(ax, sub, color, label, ls="-"):
    sub = sub.sort_values("time_s")
    ax.plot(sub.time_s, sub.mean_accuracy, color=color, lw=2.0, ls=ls, label=label)
    ax.fill_between(sub.time_s, sub.mean_accuracy - sub["sem"],
                    sub.mean_accuracy + sub["sem"], color=color, alpha=0.15, lw=0)


def _base(ax):
    annotate_stim(ax)
    ax.axhline(CHANCE, color="0.5", ls="--", lw=1.8, label="chance")
    ax.set_xlim(*XLIM); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("accuracy")
    ax.spines[["top", "right"]].set_visible(False)


# ---- expert stim pairs ------------------------------------------------------
def draw_expert_stimpair_tr(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(EXPERT)
    n = int(df.n_sessions.iloc[0]) if len(df) else 0
    _base(ax)
    for pair in ["0v1", "1v50", "0v50"]:
        sub = df[df.decoder == f"stim_{pair}"]
        if not sub.empty:
            _band(ax, sub, PAIR_COL[pair], PAIR_LAB[pair])
    ax.set_title(f"Expert stimulus-pair decoding ({n} sessions)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


# ---- reversal stim pairs (pre & post) --------------------------------------
def draw_reversal_stimpair_tr(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(AGG)
    _base(ax)
    for pair in ["0v1", "1v50", "0v50"]:
        pre = df[df.decoder == f"stim_pre_{pair}"]
        post = df[df.decoder == f"stim_post_{pair}"]
        if not pre.empty:
            _band(ax, pre, PAIR_COL[pair], PAIR_LAB[pair] + " pre", ls="-")
        if not post.empty:
            _band(ax, post, PAIR_COL[pair], PAIR_LAB[pair] + " post", ls="--")
    ax.set_title("Reversal stimulus-pair decoding (solid=pre, dashed=post)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return fig


def _reversal_stimpair_phase(phase, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(AGG)
    _base(ax)
    ls = "-" if phase == "pre" else "--"
    for pair in ["0v1", "1v50", "0v50"]:
        sub = df[df.decoder == f"stim_{phase}_{pair}"]
        if not sub.empty:
            _band(ax, sub, PAIR_COL[pair], PAIR_LAB[pair], ls=ls)
    ax.set_title(f"Reversal stimulus-pair decoding ({phase})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def draw_reversal_stimpair_tr_pre(ax=None):
    return _reversal_stimpair_phase("pre", ax)


def draw_reversal_stimpair_tr_post(ax=None):
    return _reversal_stimpair_phase("post", ax)


# ---- context / value -------------------------------------------------------
def draw_reversal_context_tr(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(AGG)
    _base(ax)
    for dec, col, lab in [("context_100to0", "#4b2362", "100%→0% cue"),
                          ("phase_50_pre_vs_post", "#c24167", "50% cue"),
                          ("context_0to100", "#edb081", "0%→100% cue")]:
        sub = df[df.decoder == dec]
        if not sub.empty:
            _band(ax, sub, col, lab)
    ax.set_title("Reversal context (pre vs post) decoding, per stimulus", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def draw_reversal_value_tr(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(AGG)
    _base(ax)
    for dec, col, lab in [("value_xor", "#1f77b4", "value (0% vs 100%)"),
                          ("stim_identity", "#d62728", "stimulus identity")]:
        sub = df[df.decoder == dec]
        if not sub.empty:
            _band(ax, sub, col, lab)
    ax.set_title("Reversal value & stimulus-identity decoding", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


# ---- cross-phase transfer (train/test) -------------------------------------
def _draw_transfer(direction, ax=None):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.5))
    df = pd.read_csv(CROSS)
    _base(ax)
    for pair, col in PAIR_COL.items():
        sub = df[(df.decoder == f"stim_cross_{pair}_{direction}") & (df.split == "test")]
        if not sub.empty:
            _band(ax, sub, col, PAIR_LAB[pair])
    dlab = "pre→post" if direction == "pre2post" else "post→pre"
    ax.set_title(f"Reversal cross-phase transfer ({dlab}, test accuracy)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return fig


def draw_reversal_transfer_pp(ax=None):
    return _draw_transfer("pre2post", ax)


def draw_reversal_transfer_pr(ax=None):
    return _draw_transfer("post2pre", ax)


def build_all(show_tag=None):
    tr = "Decoding/Reversal/Time-resolved"
    et = "Decoding/Expert/Time-resolved"
    if EXPERT.exists() and len(pd.read_csv(EXPERT)):
        save_panel(draw_expert_stimpair_tr(), et, "DEC.EXP.TR.stimpair",
                   "expert_stimpair_timeresolved", show_tag)
    if AGG.exists():
        save_panel(draw_reversal_stimpair_tr(), tr, "DEC.REV.TR.stimpair",
                   "reversal_stimpair_timeresolved", show_tag)
        save_panel(draw_reversal_stimpair_tr_pre(), tr, "DEC.REV.TR.stimpair_pre",
                   "reversal_stimpair_pre_timeresolved", show_tag)
        save_panel(draw_reversal_stimpair_tr_post(), tr, "DEC.REV.TR.stimpair_post",
                   "reversal_stimpair_post_timeresolved", show_tag)
        save_panel(draw_reversal_context_tr(), tr, "DEC.REV.TR.context",
                   "reversal_context_timeresolved", show_tag)
        save_panel(draw_reversal_value_tr(), tr, "DEC.REV.TR.value",
                   "reversal_value_timeresolved", show_tag)
    if CROSS.exists():
        save_panel(draw_reversal_transfer_pp(), tr, "DEC.REV.TR.transferPP",
                   "reversal_transfer_pre2post_timeresolved", show_tag)
        save_panel(draw_reversal_transfer_pr(), tr, "DEC.REV.TR.transferPr",
                   "reversal_transfer_post2pre_timeresolved", show_tag)


if __name__ == "__main__":
    build_all()
