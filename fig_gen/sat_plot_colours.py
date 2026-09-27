# sat_plot_colours.py
"""Stimulus colours aligned to neuronal-representations TDR notebooks.

Hex values match the printed mapping in
``analysis/18_02_26_multi_session_tdr_visualisation_stim_context_value.ipynb``
(e.g. 100%%→0%% ``#4b2362``, 50%% ``#c24167``, 0%%→100%% ``#edb081``).
Pre-reversal variants in bar/trace plots use ``lighten_hex`` for separation from post.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mc

# Notebook flare-derived mapping (100→0 purple, 50% pink, 0→100 peach)
STIM_STEM_COLOURS = {
    "100_to_0": "#4b2362",
    "50": "#c24167",
    "0_to_100": "#edb081",
}

WITHIN_PHASE_DECODER_BAR_COLOURS = {
    "0_init_100_init": "#4b2362",
    "0_init_50": "#edb081",
    "100_init_50": "#c24167",
}

# Phase-decoder line / bar colours aligned with ``vis_SAT_decoding_across_sessions``,
# ``vis_SAT_decoding_per_session``, and ``vis_SAT_decoding_accuracy_expert`` (teal, coral,
# lavender for 50% vs 100%, 0% vs 100%, 0% vs 50%).
DECODER_PREFIX_LINE_COLOURS = {
    "100_init_50": "#8dd3c7",
    "0_init_100_init": "#fb8072",
    "0_init_50": "#bebada",
}


def decoder_line_colours_for_prefixes(prefixes: Sequence[str]) -> list[str]:
    """Return hex colours for decoder prefix keys in the given order.

    Args:
        prefixes: Keys present in ``DECODER_PREFIX_LINE_COLOURS`` (e.g. main loop order).

    Returns:
        List of hex colour strings, one per prefix.
    """
    return [DECODER_PREFIX_LINE_COLOURS[p] for p in prefixes]


STEM_ORDER = ("100_to_0", "50", "0_to_100")

PRETTY_STEM_LABEL = {
    "100_to_0": "100% → 0%",
    "50": "50%",
    "0_to_100": "0% → 100%",
}


def lighten_hex(hex_color: str, amount: float = 0.55) -> str:
    """Lighten a hex colour toward white (pre-reversal / lighter emphasis).

    Args:
        hex_color: Matplotlib hex, e.g. ``'#4b2362'``.
        amount: Blend factor toward white.

    Returns:
        Hex string for the lightened colour.
    """
    r, g, b = mc.to_rgb(hex_color)

    def mix(x: float) -> float:
        return min(1.0, float(x) + float(amount) * (1.0 - x))

    return mc.to_hex((mix(r), mix(g), mix(b)))
