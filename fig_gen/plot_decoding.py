"""Time-resolved decoding figures from tidy CSVs (data/decoding).

Produces:
  decoding_overview.png     — grid of every standard decoder (mean ± SEM, chance 0.5)
  decoding_key.png          — headline decoders (stimulus / context / value) overlaid
  decoding_cross_phase.png  — cross-phase generalisation (train vs test)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from _common import OUTDIR, DATA, load, annotate_stim

CHANCE = 0.5
XLIM = (-1.0, 3.0)


def _plot_band(ax, sub, color, label):
    sub = sub.sort_values("time_s")
    ax.plot(sub.time_s, sub.mean_accuracy, color=color, lw=2.0, label=label)
    ax.fill_between(sub.time_s, sub.mean_accuracy - sub["sem"],
                    sub.mean_accuracy + sub["sem"], color=color, alpha=0.18, lw=0)


def plot_overview():
    df = load("decoding/decoding_aggregated.csv")
    decoders = list(dict.fromkeys(df.decoder))
    ncols = 3
    nrows = -(-len(decoders) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, dec in enumerate(decoders):
        ax = axes[i // ncols][i % ncols]
        annotate_stim(ax)
        ax.axhline(CHANCE, color="0.5", ls="--", lw=0.8)
        _plot_band(ax, df[df.decoder == dec], "#2166ac", dec)
        ax.set_title(dec, fontsize=8)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0.35, 0.9)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(len(decoders), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel("Time from stimulus onset (s)")
    fig.supylabel("Decoding accuracy")
    fig.suptitle("Time-resolved decoding — all problems (mean ± SEM across sessions)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "decoding_overview.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "decoding_overview.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_key():
    df = load("decoding/decoding_aggregated.csv")
    picks = [
        ("value_xor", "#c24167", "value (XOR)"),
        ("context_pooled", "#4b2362", "context (pooled)"),
        ("stim_post_0v1", "#edb081", "stim 0%vs100% post"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    annotate_stim(ax)
    ax.axhline(CHANCE, color="0.5", ls="--", lw=0.9, label="chance")
    for dec, color, label in picks:
        sub = df[df.decoder == dec]
        if not sub.empty:
            _plot_band(ax, sub, color, label)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.4, 0.9)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_title("Headline decoders")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "decoding_key.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "decoding_key.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_stim_vs_value():
    """The two complementary decoders over the same 4 (cue x context) cells:
    stimulus identity vs value (XOR)."""
    df = load("decoding/decoding_aggregated.csv")
    picks = [
        ("stim_identity", "#2ca02c", "stimulus identity (100→0 vs 0→100 cue)"),
        ("value_xor", "#c24167", "value (0% vs 100%, pooled)"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    annotate_stim(ax)
    ax.axhline(CHANCE, color="0.5", ls="--", lw=0.9, label="chance")
    for dec, color, label in picks:
        sub = df[df.decoder == dec]
        if not sub.empty:
            _plot_band(ax, sub, color, label)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.4, 0.9)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_title("Stimulus identity vs value (complementary decoders)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "decoding_stim_vs_value.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "decoding_stim_vs_value.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_cross_phase():
    df = load("decoding/decoding_cross_phase.csv")
    if df.empty:
        return
    decoders = list(dict.fromkeys(df.decoder))
    ncols = 3
    nrows = -(-len(decoders) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, dec in enumerate(decoders):
        ax = axes[i // ncols][i % ncols]
        annotate_stim(ax)
        ax.axhline(CHANCE, color="0.5", ls="--", lw=0.8)
        for split, color in [("train", "#2166ac"), ("test", "#b2182b")]:
            sub = df[(df.decoder == dec) & (df.split == split)].sort_values("time_s")
            if sub.empty:
                continue
            ax.plot(sub.time_s, sub.mean_accuracy, color=color, lw=1.8, label=split)
            ax.fill_between(sub.time_s, sub.mean_accuracy - sub["sem"],
                            sub.mean_accuracy + sub["sem"], color=color, alpha=0.15, lw=0)
        ax.set_title(dec, fontsize=8)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0.35, 0.9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    for j in range(len(decoders), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.supxlabel("Time from stimulus onset (s)")
    fig.supylabel("Decoding accuracy")
    fig.suptitle("Cross-phase stimulus generalisation (train vs test)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / "decoding_cross_phase.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "decoding_cross_phase.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def plot_expert():
    """Expert (non-reversal) pairwise stimulus decoders."""
    path = DATA / "decoding" / "decoding_expert.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    colours = {"stim_0v1": "#fb8072", "stim_0v50": "#bebada", "stim_1v50": "#8dd3c7"}
    labels = {"stim_0v1": "0% vs 100%", "stim_0v50": "0% vs 50%", "stim_1v50": "100% vs 50%"}
    n = int(df.n_sessions.iloc[0]) if len(df) else 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    annotate_stim(ax)
    ax.axhline(CHANCE, color="0.5", ls="--", lw=0.9, label="chance")
    for dec in ["stim_0v1", "stim_1v50", "stim_0v50"]:
        sub = df[df.decoder == dec]
        if not sub.empty:
            _plot_band(ax, sub, colours[dec], labels[dec])
    ax.set_xlim(*XLIM)
    ax.set_ylim(0.4, 0.9)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Decoding accuracy")
    ax.set_title(f"Expert stimulus-pair decoding ({n} sessions)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTDIR / "decoding_expert.png", dpi=150, bbox_inches="tight")
    fig.savefig(str(OUTDIR / "decoding_expert.png").replace(".png", ".pdf"), bbox_inches="tight")   # vector companion
    plt.close(fig)


def main():
    plot_overview()
    plot_key()
    plot_stim_vs_value()
    plot_cross_phase()
    plot_expert()
    print("Decoding figures →", OUTDIR)


if __name__ == "__main__":
    main()
