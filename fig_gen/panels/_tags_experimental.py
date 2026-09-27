"""Modular figure-panel framework with a toggleable tag system.

Each individual figure is drawn by a small self-contained ``draw_*(ax=...)``
function so it can either be saved as a standalone panel or dropped into a
larger multi-panel figure later (just pass an existing Axes).

Every panel has a short hierarchical **tag** (e.g. ``DEC.REV.TP.value``) that is
(1) put in the output filename and (2) stamped on the figure itself. The stamp
can be turned off globally (``SHOW_TAG = False``) or per call.

Folder layout mirrors the requested hierarchy under ``figures/``:
    Behaviour/Expert, Behaviour/Reversal,
    Population activity/Expert, Population activity/Reversal,
    Subgroups/Expert, Subgroups/Reversal,
    Decoding/{Expert,Reversal}/{Time-pooled,Time-resolved},
    Dimensionality/{Expert,Reversal}[/PCA][/TDR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# shared stylesheet (large axis labels, no spines, borderless bars, …)
plt.style.use(str(Path(__file__).resolve().parent / "transfer.mplstyle"))

# make figures/_common importable from panel modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIG_ROOT = Path(__file__).resolve().parent.parent        # .../figures
OUT_ROOT = FIG_ROOT / "figs"                              # .../figures/figs (all generated output)
DATA = FIG_ROOT.parent / "data"

SHOW_TAG = True   # global default; individual saves can override


def _capitalise(label):
    return (label[0].upper() + label[1:]) if label else label


def finalize_axes(fig, force_remove_titles=False):
    """Apply shared conventions to every axes in a figure:
      * capitalise x/y axis labels,
      * drop axes titles on single-panel figures and composites, but KEEP them on
        heatmap axes (with an image) and on multi-panel per-session GRIDS where the
        title is the session identifier.
    (Figure-level suptitles are left untouched.)
    """
    single = len(fig.axes) <= 2            # one main axes (+ optional colorbar)
    for ax in fig.axes:
        remove = (force_remove_titles or single) and not ax.get_images()
        if remove:
            ax.set_title("")
        ax.set_xlabel(_capitalise(ax.get_xlabel()))
        ax.set_ylabel(_capitalise(ax.get_ylabel()))


def new_panel(ax=None, figsize=(6.0, 4.0)):
    """Return (fig, ax, owns_fig). If ax is None a new fig/ax is created."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def stamp_tag(fig, tag, show=None):
    show = SHOW_TAG if show is None else show
    if show and tag:
        fig.text(0.006, 0.994, tag, ha="left", va="top", fontsize=7,
                 color="0.4", family="monospace", zorder=1000)


def _norm_folder(folder):
    """lower-case, underscores; no spaces or hyphens (keeps '/' separators)."""
    parts = folder.split("/")
    return "/".join(p.lower().replace(" ", "_").replace("-", "_") for p in parts)


def _norm_tag(tag):
    """No dots in figure codes (bad in filenames) — use underscores."""
    return tag.replace(".", "_")


def save_panel(fig, folder, tag, name, show_tag=None):
    """Save a figure into figures/<folder>/<tag>__<name>.png with its tag.

    Folder names are lower_cased with underscores; tags have dots replaced by
    underscores (both filename and the on-figure stamp use the normalised tag).
    """
    folder = _norm_folder(folder)
    tag = _norm_tag(tag)
    out_dir = OUT_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    finalize_axes(fig)
    stamp_tag(fig, tag, show_tag)
    path = out_dir / f"{tag}__{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")   # vector companion, every panel
    plt.close(fig)
    print(f"  {tag:24} -> {folder}/{path.name} (+ .pdf)")
    return path
