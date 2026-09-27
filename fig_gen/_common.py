"""Shared helpers for the transfer figure scripts.

The figure scripts read ONLY the tidy CSVs in ``../data`` — they never touch the
original pickles or the analysis codebase. This module centralises paths,
house-style stimulus colours (via ``sat_plot_colours``) and light smoothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE / "figs" / "output"
OUTDIR.mkdir(exist_ok=True)

try:
    from sat_plot_colours import STIM_STEM_COLOURS, lighten_hex
except Exception:  # pragma: no cover - fallback if helper missing
    STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}

    def lighten_hex(h, amount=0.45):
        import matplotlib.colors as mc
        r, g, b = mc.to_rgb(h)
        mix = lambda x: min(1.0, x + amount * (1.0 - x))
        return mc.to_hex((mix(r), mix(g), mix(b)))


def _stem_key(condition: str) -> str:
    if "100%-->0%" in condition or "100_to_0" in condition:
        return "100_to_0"
    if "0%-->100%" in condition or "0_to_100" in condition:
        return "0_to_100"
    return "50"


def condition_style(condition: str) -> dict:
    """Return plot kwargs (color, linestyle, lw, label) for a condition label.

    Pre-reversal = solid full-saturation; post-reversal = dashed lightened.
    """
    stem = _stem_key(condition)
    base = STIM_STEM_COLOURS[stem]
    is_post = "post" in condition.lower()
    pretty = {"100_to_0": "100%→0%", "50": "50%", "0_to_100": "0%→100%"}[stem]
    if is_post:
        return dict(color=lighten_hex(base, 0.45), linestyle="--", lw=2.0,
                    label=f"{pretty}, post")
    if "pre" in condition.lower():
        return dict(color=base, linestyle="-", lw=2.0, label=f"{pretty}, pre")
    return dict(color=base, linestyle="-", lw=2.0, label=pretty)


# Canonical condition order for legends
CONDITION_ORDER = [
    "100%-->0% (pre-Rev)", "50% (pre-Rev)", "0%-->100% (pre-Rev)",
    "100%-->0% (post-Rev)", "50% (post-Rev)", "0%-->100% (post-Rev)",
]


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    """Simple centred moving-average smoothing (win in samples). win<=1 -> no-op."""
    y = np.asarray(y, dtype=float)
    if win <= 1:
        return y
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(y, size=int(win))


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def annotate_stim(ax, dur_s: float = 2.0):
    """Mark the stimulus window [0, dur_s] with vertical lines (onset + offset)."""
    ax.axvline(0, color="0.4", lw=1.8, ls=":", label="stimulus on/off")
    ax.axvline(dur_s, color="0.4", lw=1.8, ls=":")
