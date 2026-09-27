"""Anticipatory lick rate over trials, one trace per stimulus type, per session.

Consumes data/lick/lick_rates_per_trial.csv (produced by extract/extract_lick.py,
or by any pipeline that writes the same columns):
    session, trial_number, stimulus_label, anticipatory_lick_rate

Produces:
  lick_rate_over_trials.png   small-multiples grid, one panel per session, three
                              stimulus traces (100%->0%, 50%, 0%->100%).

If the CSV is absent (raw lick files not yet available), the script prints a
clear message and exits without error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import OUTDIR, DATA
try:
    from sat_plot_colours import STIM_STEM_COLOURS
except Exception:
    STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}

LABEL_ORDER = ["100_to_0", "50", "0_to_100"]
PRETTY = {"100_to_0": "100%→0%", "50": "50%", "0_to_100": "0%→100%"}

# Expert sessions: fixed reward-probability stimuli (labels 100/50/0), coloured
# by reward-probability level (100% purple, 50% pink, 0% peach).
EXPERT_ORDER = ["100", "50", "0"]
EXPERT_PRETTY = {"100": "100%", "50": "50%", "0": "0%"}
EXPERT_COLOURS = {"100": STIM_STEM_COLOURS["100_to_0"],
                  "50": STIM_STEM_COLOURS["50"],
                  "0": STIM_STEM_COLOURS["0_to_100"]}

ROLL = 15  # rolling-mean window over trials (smoothing); set 1 to disable
BIN = 15   # bin width (trials) for the across-session combined plot
MIN_SESSIONS = 5  # min sessions contributing to a bin to plot it (combined)

CSV = DATA / "lick" / "lick_rates_per_trial.csv"
META = DATA / "single_trial" / "single_trial_metadata.csv"


def reversal_trials():
    """Reversal trial per session = boundary between the last pre-Rev and first
    post-Rev trial, read from the single-trial metadata (pre-Rev/post-Rev
    condition labels). Returns {session: reversal_x} or {} if unavailable."""
    if not META.exists():
        return {}
    m = pd.read_csv(META)
    out = {}
    for sess, g in m.groupby("session"):
        pre = g[g.condition.str.contains("pre-Rev")].session_trial_number
        post = g[g.condition.str.contains("post-Rev")].session_trial_number
        if len(pre) and len(post):
            out[sess] = 0.5 * (pre.max() + post.min())
        elif len(post):
            out[sess] = post.min()
    return out


def plot(csv=CSV, out_name="lick_rate_over_trials.png", draw_reversal=True,
         title_extra="", label_order=None, pretty=None, colours=None):
    label_order = label_order or LABEL_ORDER
    pretty = pretty or PRETTY
    colours = colours or STIM_STEM_COLOURS
    df = pd.read_csv(csv)
    revs = reversal_trials() if draw_reversal else {}
    sessions = sorted(df.session.unique())
    ncols = 4
    nrows = -(-len(sessions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                             squeeze=False, sharex=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sdf = df[df.session == sess]
        for lab in label_order:
            s = sdf[sdf.stimulus_label.astype(str) == lab].sort_values("trial_number")
            if s.empty:
                continue
            y = s.anticipatory_lick_rate.rolling(ROLL, min_periods=1, center=True).mean()
            ax.plot(s.trial_number, y, color=colours[lab], lw=1.6,
                    label=pretty[lab])
        if sess in revs:
            ax.axvline(revs[sess], color="0.35", ls="--", lw=1.2,
                       label="reversal")
        ax.set_title(sess, fontsize=8)
        ax.set_xlabel("trial", fontsize=8)
        ax.set_ylabel("lick rate (Hz)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(len(sessions), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(fontsize=7, frameon=False)
    smooth_txt = (f"{ROLL}-trial rolling mean" if ROLL > 1 else "raw, unsmoothed")
    rev_txt = "dashed = reversal; " if draw_reversal else ""
    fig.suptitle("Anticipatory lick rate over trials, by stimulus type "
                 f"{title_extra}({rev_txt}{smooth_txt})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / out_name, dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / out_name).replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)
    print("Lick figure →", OUTDIR / out_name)


def plot_expert_bar(csv, out_name="lick_rate_expert_bar.png"):
    """Bar plot of mean anticipatory lick rate per stimulus (expert, 3 stimuli).

    Bars = mean over sessions of each session's mean lick rate per stimulus;
    error bars = SEM across sessions; grey points = per-session means.
    """
    df = pd.read_csv(csv)
    per_sess = (df.groupby(["stimulus_label", "session"])
                  .anticipatory_lick_rate.mean().reset_index())
    order = [l for l in EXPERT_ORDER if l in per_sess.stimulus_label.astype(str).unique()]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for x, lab in enumerate(order):
        vals = per_sess[per_sess.stimulus_label.astype(str) == lab].anticipatory_lick_rate.values
        m = np.nanmean(vals)
        se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ax.bar(x, m, width=0.65, color=EXPERT_COLOURS[lab], edgecolor="k", lw=0.6,
               yerr=se, capsize=4, zorder=2)
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.25
        ax.scatter(x + jitter, vals, color="0.3", s=14, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([EXPERT_PRETTY[l] for l in order])
    ax.set_xlabel("stimulus (reward probability)")
    ax.set_ylabel("anticipatory lick rate (Hz)")
    n = per_sess.session.nunique()
    ax.set_title(f"Expert anticipatory lick rate by stimulus\n(mean ± SEM over {n} sessions)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / out_name, dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / out_name).replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)
    print("Expert lick bar plot →", OUTDIR / out_name)


def _placeholder():
    """Write a clearly-labelled placeholder when no lick data is present yet."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5,
            "Anticipatory lick rate — awaiting raw lick data.\n\n"
            "The lick_rates_per_trial.csv here is an empty schema template.\n"
            "Run extract/extract_lick.py with the raw lick files connected\n"
            "(or drop in a CSV with the same columns), then re-run this script.",
            ha="center", va="center", fontsize=11)
    fig.savefig(OUTDIR / "lick_rate_over_trials.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "lick_rate_over_trials.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)
    print("[plot_lick] no lick data yet — wrote placeholder figure.")


def plot_combined():
    """Across-session summary: lick rate vs trial relative to each session's
    reversal, averaged over sessions per stimulus type (mean ± SEM)."""
    df = pd.read_csv(CSV)
    revs = reversal_trials()
    df = df[df.session.isin(revs)].copy()
    if df.empty:
        return
    # trials relative to reversal (reversal boundary at 0), binned
    df["rel"] = df.trial_number - df.session.map(revs)
    df["bin"] = np.floor(df.rel / BIN) * BIN + BIN / 2.0

    # per-session bin means, then mean ± SEM across sessions
    per = (df.groupby(["stimulus_label", "bin", "session"])
             .anticipatory_lick_rate.mean().reset_index())
    agg = (per.groupby(["stimulus_label", "bin"])
              .anticipatory_lick_rate.agg(["mean", "sem", "count"]).reset_index())
    agg = agg[agg["count"] >= MIN_SESSIONS]

    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in LABEL_ORDER:
        s = agg[agg.stimulus_label == lab].sort_values("bin")
        if s.empty:
            continue
        ax.plot(s.bin, s["mean"], color=STIM_STEM_COLOURS[lab], lw=2.2,
                label=PRETTY[lab])
        ax.fill_between(s.bin, s["mean"] - s["sem"], s["mean"] + s["sem"],
                        color=STIM_STEM_COLOURS[lab], alpha=0.2, lw=0)
    ax.axvline(0, color="0.35", ls="--", lw=1.4, label="reversal")
    ax.set_xlabel("trial relative to reversal")
    ax.set_ylabel("anticipatory lick rate (Hz)")
    ax.set_title(f"Across-session anticipatory lick rate (mean ± SEM over "
                 f"{df.session.nunique()} sessions; {BIN}-trial bins)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "lick_rate_combined.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "lick_rate_combined.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)
    print("Combined lick figure →", OUTDIR / "lick_rate_combined.png")


def main():
    # Reversal-session licks (with reversal marker + reversal-aligned combined plot)
    if CSV.exists() and len(pd.read_csv(CSV)):
        plot()
        plot_combined()
    else:
        print(f"[plot_lick] {CSV} not found/empty — run extract/extract_lick.py first.")
        _placeholder()

    # Expert-session licks (no reversal): plotted if present. Produce with
    #   TRANSFER_LICK_SESSION_SET=expert python extract/extract_lick.py
    expert_csv = DATA / "lick" / "lick_rates_per_trial_expert.csv"
    if expert_csv.exists() and len(pd.read_csv(expert_csv)):
        plot(csv=expert_csv, out_name="lick_rate_over_trials_expert.png",
             draw_reversal=False, title_extra="— expert sessions ",
             label_order=EXPERT_ORDER, pretty=EXPERT_PRETTY, colours=EXPERT_COLOURS)
        plot_expert_bar(expert_csv)
    else:
        for out_name in ("lick_rate_expert_bar.png", "lick_rate_over_trials_expert.png"):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            ax.text(0.5, 0.5,
                    "Expert lick data not present yet.\n\n"
                    "Run:  TRANSFER_LICK_SESSION_SET=expert python extract/extract_lick.py\n"
                    "then re-run this script.",
                    ha="center", va="center", fontsize=11)
            fig.savefig(OUTDIR / out_name, dpi=150, bbox_inches="tight")
            fig.savefig(str(OUTDIR / out_name).replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
            plt.close(fig)
        print("[plot_lick] no expert lick data — wrote expert placeholders.")


if __name__ == "__main__":
    main()
