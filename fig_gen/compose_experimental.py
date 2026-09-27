"""Compose individual panels into multi-panel figures.

Each subpanel is drawn by its own ``draw_*(ax=…)`` function (from panels/*.py)
onto a shared gridspec, and labelled with its panel tag (toggle with SHOW_TAG).
Outputs go to figures/composites/<TAG>__<name>.png.

    python compose.py
    SHOW_TAG=0 python compose.py   # hide the per-panel tag labels
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "panels"))

import _tags
_tags.SHOW_TAG = os.environ.get("SHOW_TAG", "1") not in ("0", "false", "False")
from _tags import FIG_ROOT, OUT_ROOT, finalize_axes  # noqa: E402

import behaviour, population, subgroups, decoding, decoding_pooled, reversal_gradient, dimensionality  # noqa: E402

OUT = OUT_ROOT / "composites"
OUT.mkdir(parents=True, exist_ok=True)

EXP_HEAT = "2025-07-18_1_SAT037"   # SAT037 2025-07-18 (expert)
REV_HEAT = "2025-07-31_2_SAT037"   # SAT037 2025-07-31 (reversal)


import string  # noqa: E402


def _panel(fig, gs, r, c, tag, fn, letter=None):
    ax = fig.add_subplot(gs[r, c])
    fn(ax=ax)
    if letter is not None:
        ax.text(-0.1, 1.22, letter, transform=ax.transAxes, fontsize=30,
                fontweight="bold", va="top", ha="left")
    elif _tags.SHOW_TAG:
        ax.text(0.015, 0.985, tag, transform=ax.transAxes, fontsize=6, va="top",
                ha="left", color="0.35", family="monospace", zorder=1000)
    return ax


def _build_grid(fig, gs, rows, letters):
    """Draw a list of rows (each a list of (tag, fn)); letters run row-major."""
    idx = 0
    for r, row in enumerate(rows):
        for c, (tag, fn) in enumerate(row):
            _panel(fig, gs, r, c, tag, fn,
                   letter=string.ascii_lowercase[idx] if letters else None)
            idx += 1


# expert stimulus labels name the same visual stimulus as the reversal cues,
# so collapse them into one legend entry.
_CANON = {"100%": "100%→0%", "0%": "0%→100%"}   # 50% is already shared


def _combine_legends(fig, exempt=None):
    """Strip each panel's own legend and build ONE combined legend to the right
    of the panel grid (deduped by label, so it can be read for all panels).
    ``exempt`` (a list of Axes) is left alone entirely -- its own in-panel
    legend stays put and isn't folded into the combined one (used for FIG2's
    panel 'a', whose pre/post legend the composite keeps local rather than
    shared)."""
    exempt = exempt or []
    seen = {}
    for ax in fig.axes:
        if ax in exempt:
            continue
        h, l = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for hh, ll in zip(h, l):
            ll = _CANON.get(ll, ll)
            if ll and not ll.startswith("_") and ll not in seen:
                seen[ll] = hh
    if seen:
        fig.legend(seen.values(), seen.keys(), loc="center left",
                   bbox_to_anchor=(0.94, 0.5), frameon=False, fontsize=15)


def _strip_all_legends(fig, exempt=None):
    """No-legend variant: remove every axes' own legend outright (no shared
    figure-level legend either). ``exempt`` axes keep their own legend."""
    exempt = exempt or []
    for ax in fig.axes:
        if ax in exempt:
            continue
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def _save(fig, tag, name, legend=True, exempt_legend_axes=None, subdir=None):
    if legend:
        _combine_legends(fig, exempt=exempt_legend_axes)
    else:
        _strip_all_legends(fig, exempt=exempt_legend_axes)
    finalize_axes(fig, force_remove_titles=True)
    out_dir = (OUT / subdir) if subdir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}__{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")   # vector companion, every composite
    plt.close(fig)
    rel = f"composites/{subdir}/{path.name}" if subdir else f"composites/{path.name}"
    print(f"  {tag:12} -> {rel} (+ .pdf)")


def figure1(letters=False, exclude_nonresp=False, legend=True, method="temporal"):
    """Expert (top) vs reversal (bottom) cross-analysis overview.

    When ``exclude_nonresp`` is True, panels c (expert sizes), h (reversal
    sizes) and i (reversal Sankey) drop non-responding cells.

    ``legend=False`` produces a no-legend variant (every panel's own legend,
    plus the combined figure-level one, dropped -- see _strip_all_legends).

    ``method``: "temporal" or "time_averaged" -- sets subgroups.METHOD (which
    all subgroups.* panels here read at call time) for the duration of this
    build, so every panel below that touches responder groups (SUB_EXP_sizes,
    SUB_REV_sizes, SUB_REV_sankeycomb, both heatmappools) uses the matching
    responder-significance criterion. BEH/POP/DEC panels are unaffected.

    NOTE: previously the two heatmappool panels (e, j) ignored
    ``exclude_nonresp`` entirely -- they kept drawing every cell with a
    divider line even in the "responders only" variant, inconsistent with
    the sizes/Sankey panels alongside them. Both now honour it too.
    """
    inc = not exclude_nonresp
    _orig_method = subgroups.METHOD
    subgroups.METHOD = method
    try:
        top = [
            ("BEH_EXP_lickbar", behaviour.draw_expert_lick_bar),
            ("POP_EXP_mean", population.draw_expert_mean),
            ("SUB_EXP_sizes", lambda ax: subgroups.draw_expert_sizes(ax, include_nonresp=inc)),
            ("DEC_EXP_TP_stimpair", decoding_pooled.draw_expert_stimpair_bar),
            ("SUB_EXP_heatmappool", lambda ax: subgroups.draw_pooled_heatmap("expert", ax, exclude_nonresp=exclude_nonresp)),
        ]
        bottom = [
            ("BEH_REV_lickcombined", behaviour.draw_reversal_lick_combined),
            ("POP_REV_mean", population.draw_reversal_mean),
            ("SUB_REV_sizes", lambda ax: subgroups.draw_reversal_sizes(ax, include_nonresp=inc)),
            ("SUB_REV_sankeycomb", lambda ax: subgroups.draw_sankey_combined(ax, exclude_nonresp=exclude_nonresp)),
            ("SUB_REV_heatmappool", lambda ax: subgroups.draw_pooled_heatmap("reversal", ax, exclude_nonresp=exclude_nonresp)),
        ]
        fig = plt.figure(figsize=(6.4 * 5, 10.5))
        # Column 2 ("sizes" bar, needs room for its x-tick labels) and column 3
        # ("SUB_REV_sankeycomb" in the bottom row -- the Sankey looked small
        # relative to everything else at 1x) both get extra width.
        gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.6, wspace=0.42,
                               width_ratios=[1, 1, 1.3, 1.45, 1], top=0.94)
        _build_grid(fig, gs, [top, bottom], letters)
        suffix = (("_responders" if exclude_nonresp else "") + ("_lettered" if letters else "")
                  + ("_nolegend" if not legend else "") + f"_{method}")
        _save(fig, "FIG1", "expert_reversal_overview" + suffix, legend=legend, subdir=method)
    finally:
        subgroups.METHOD = _orig_method


def figure1_gradient(letters=False, exclude_nonresp=False, legend=True, method="temporal", window=100):
    """Same as figure1(), but panel f (BEH_REV_lickcombined -- the full
    trial-resolved reversal lick-rate curve) is replaced with
    reversal_gradient.draw_reversal_gradient_bars: the per-stimulus fixed-
    window learning-rate GRADIENT as one bar each, instead of the full
    curve (see chat: "quantify the gradient of the lick rate vs trials...
    post reversal ... so we can compare learning rates without having to
    look at the full trial-resolved curves"). Every other panel, and every
    parameter, is identical to figure1() -- see that function's docstring.
    ``window`` (default 100 trials) is passed straight through to
    draw_reversal_gradient_bars.
    """
    inc = not exclude_nonresp
    _orig_method = subgroups.METHOD
    subgroups.METHOD = method
    try:
        top = [
            ("BEH_EXP_lickbar", behaviour.draw_expert_lick_bar),
            ("POP_EXP_mean", population.draw_expert_mean),
            ("SUB_EXP_sizes", lambda ax: subgroups.draw_expert_sizes(ax, include_nonresp=inc)),
            ("DEC_EXP_TP_stimpair", decoding_pooled.draw_expert_stimpair_bar),
            ("SUB_EXP_heatmappool", lambda ax: subgroups.draw_pooled_heatmap("expert", ax, exclude_nonresp=exclude_nonresp)),
        ]
        bottom = [
            ("BEH_REV_gradient", lambda ax: reversal_gradient.draw_reversal_gradient_bars(window=window, ax=ax)),
            ("POP_REV_mean", population.draw_reversal_mean),
            ("SUB_REV_sizes", lambda ax: subgroups.draw_reversal_sizes(ax, include_nonresp=inc)),
            ("SUB_REV_sankeycomb", lambda ax: subgroups.draw_sankey_combined(ax, exclude_nonresp=exclude_nonresp)),
            ("SUB_REV_heatmappool", lambda ax: subgroups.draw_pooled_heatmap("reversal", ax, exclude_nonresp=exclude_nonresp)),
        ]
        fig = plt.figure(figsize=(6.4 * 5, 10.5))
        gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.6, wspace=0.42,
                               width_ratios=[1, 1, 1.3, 1.45, 1], top=0.94)
        _build_grid(fig, gs, [top, bottom], letters)
        suffix = (("_responders" if exclude_nonresp else "") + ("_lettered" if letters else "")
                  + ("_nolegend" if not legend else "") + f"_{method}")
        _save(fig, "FIG1G", "expert_reversal_overview_gradient" + suffix, legend=legend, subdir=method)
    finally:
        subgroups.METHOD = _orig_method


def figure_gradient_fit_diagnostic(window=100):
    """Supplementary: reversal_gradient.draw_fit_diagnostic_grid() as its own
    saved figure -- every reversal session's own fit (points + OLS line),
    not just the resulting slope bar (see figure1_gradient's BEH_REV_gradient
    panel for that). Its own function (not a _panel()/gridspec composite)
    since draw_fit_diagnostic_grid already builds and returns its own
    complete multi-session grid figure."""
    fig = reversal_gradient.draw_fit_diagnostic_grid(window=window)
    out_dir = OUT / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_ = out_dir / f"GRADIENT_FITDIAG_w{window}__reversal_gradient_fit_diagnostic_w{window}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight")
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  GRADIENT_FITDIAG_w{window} -> supplementary/{path_.name} (+ .pdf)")


def figure_gradient_cross_session_fit(window=100):
    """Supplementary: reversal_gradient.draw_cross_session_fit() as its own
    saved figure -- the gradient fit to the CROSS-SESSION AVERAGE lick-rate
    curve (same session-inclusion floor + smoothing as behaviour.py's
    draw_reversal_lick_combined), alongside figure_gradient_fit_diagnostic's
    per-session fits."""
    fig = reversal_gradient.draw_cross_session_fit(window=window)
    out_dir = OUT / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_ = out_dir / f"GRADIENT_CROSSSESSION_w{window}__reversal_gradient_cross_session_fit_w{window}.png"
    fig.savefig(path_, dpi=150, bbox_inches="tight")
    fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  GRADIENT_CROSSSESSION_w{window} -> supplementary/{path_.name} (+ .pdf)")


def figure_dimensionality_grids():
    """Supplementary: dimensionality.py's 3 multi-session grid figures
    (reversal PCA -- all 13 sessions, expert PCA -- representative 12-of-43
    subset, reversal TDR -- all 13 sessions) as their own saved figures --
    each already builds and returns its own complete grid, so (like
    figure_gradient_fit_diagnostic() above) this just saves them rather than
    building a _panel()/gridspec composite."""
    out_dir = OUT / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    proj = pd.read_csv(dimensionality.PROJ)
    expert_sessions = sorted(proj[proj.scope == "expert"].session.unique())
    figs = [
        ("DIM_PCA_GRID_reversal", "reversal_pca_grid_all_sessions",
         dimensionality.draw_pca_projections_grid("reversal")),
        ("DIM_PCA_GRID_expert", "expert_pca_grid_subset",
         dimensionality.draw_pca_projections_grid(
             "expert", sessions=dimensionality._representative_subset(expert_sessions))),
        ("DIM_TDR_GRID_reversal", "reversal_tdr_grid_all_sessions",
         dimensionality.draw_tdr_projections_grid()),
    ]
    for tag, name, fig in figs:
        path_ = out_dir / f"{tag}__{name}.png"
        fig.savefig(path_, dpi=150, bbox_inches="tight")
        fig.savefig(path_.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  {tag} -> supplementary/{path_.name} (+ .pdf)")


def figure2(letters=False, legend=True):
    """Reversal decoding: time-pooled (top) and time-resolved (bottom).

    ``legend=False`` drops every OTHER panel's legend/the combined
    figure-level legend, but panel 'a' (DEC_REV_TP_stimpair) always keeps
    its own in-panel pre/post legend regardless -- kept local (not swept
    into the shared legend) in the ``legend=True`` variant too.
    """
    top = [
        ("DEC_REV_TP_stimpair", decoding_pooled.draw_reversal_stimpair_bar),
        ("DEC_REV_TP_transferPP", decoding_pooled.draw_reversal_transfer_pp),
        ("DEC_REV_TP_context", decoding_pooled.draw_reversal_context_bar),
        ("DEC_REV_TP_value", decoding_pooled.draw_reversal_value_bar),
    ]
    bottom = [
        ("DEC_REV_TR_stimpair_pre", decoding.draw_reversal_stimpair_tr_pre),
        ("DEC_REV_TR_stimpair_post", decoding.draw_reversal_stimpair_tr_post),
        ("DEC_REV_TR_context", decoding.draw_reversal_context_tr),
        ("DEC_REV_TP_stimidentity", decoding_pooled.draw_reversal_stimidentity_bar),
    ]
    fig = plt.figure(figsize=(6.6 * 4, 9.5))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4,
                           width_ratios=[1, 1, 1, 0.6], top=0.95)
    _build_grid(fig, gs, [top, bottom], letters)
    panel_a_exempt = [fig.axes[0]]   # DEC_REV_TP_stimpair -- keeps its own pre/post legend always
    suffix = ("_lettered" if letters else "") + ("_nolegend" if not legend else "")
    _save(fig, "FIG2", "reversal_decoding" + suffix, legend=legend, exempt_legend_axes=panel_a_exempt)


def main():
    for legend in (True, False):
        for method in ("temporal", "time_averaged"):
            figure1(letters=False, legend=legend, method=method)
            figure1(letters=True, legend=legend, method=method)
            figure1(letters=False, exclude_nonresp=True, legend=legend, method=method)
            figure1(letters=True, exclude_nonresp=True, legend=legend, method=method)
            figure1_gradient(letters=False, legend=legend, method=method)
            figure1_gradient(letters=True, legend=legend, method=method)
            figure1_gradient(letters=False, exclude_nonresp=True, legend=legend, method=method)
            figure1_gradient(letters=True, exclude_nonresp=True, legend=legend, method=method)
        figure2(letters=False, legend=legend); figure2(letters=True, legend=legend)
    figure_gradient_fit_diagnostic(window=100)
    figure_gradient_fit_diagnostic(window=200)
    figure_gradient_cross_session_fit(window=100)
    figure_gradient_cross_session_fit(window=200)
    figure_dimensionality_grids()
    print("Composites →", OUT)


if __name__ == "__main__":
    main()
