"""Time-resolved PCA state-space panels -- RNN models.

Draws the two figure types requested for the RNN side, mirroring the real-
data pca_phase_space (tdr_phase_space.png-style) and pca_pc_timecourses.png
figures in neuronal-representations/results/transfer/figures/plot_pca.py.
The actual PCA fit + per-seed aggregation lives in ../analysis/
population_pca.py -- this module only draws.

  draw_pca_phase_space(model_type, pcs=(0,1), ax=None)
      2D trajectory through PC space, one line per stimulus, solid = pre-
      reversal, dashed = post-reversal, ● start / ■ end markers, restricted
      to the post-stimulus-onset window (t >= 0) -- same convention as the
      real repo's dimensionality.py masking (m = time_s >= 0).

  draw_pca_timecourses(model_type, n_components=3)
      One stacked subplot per PC, full trial timecourse (including the ITI
      baseline), pre solid / post dashed, with dotted lines at the ITI|stim
      and stim|outcome boundaries (same convention as figures.py's
      group_grid).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "style"))
sys.path.insert(0, str(_HERE.parent / "transfer" / "code"))
sys.path.insert(0, str(_HERE.parent / "analysis"))

from _tags import new_panel, save_panel  # noqa: E402
import style as S  # noqa: E402
import figures as F  # noqa: E402
import population_pca as PPCA  # noqa: E402  (combined/analysis/population_pca.py)

MODEL_TYPES = PPCA.MODEL_TYPES
STIM_ORDER = PPCA.STIM_ORDER
STIM_PRETTY = {"0": "0%", "50": "50%", "100": "100%"}


def draw_pca_phase_space(model_type, pcs=(0, 1), ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.8))
    D = PPCA.population_pca_trajectories(model_type)
    t = D["t"]
    m = t >= 0     # drop the ITI baseline, matching the real repo's t>=0 window
    xi, yi = pcs
    for si, stim in enumerate(STIM_ORDER):
        colour = S.STIM_COLOURS[stim]
        for mean, ls, lw, marker_end in ((D["pre_mean"], "-", 1.8, "s"),
                                          (D["post_mean"], "--", 1.8, "s")):
            x, y = mean[m, si, xi], mean[m, si, yi]
            ax.plot(x, y, color=colour, ls=ls, lw=lw)
            ax.scatter(x[-1], y[-1], color=colour, s=32, marker=marker_end,
                       edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(D["pre_mean"][m, si, xi][0], D["pre_mean"][m, si, yi][0],
                   color=colour, s=24, zorder=3)
    ve = D["var_explained"]
    ax.text(0.02, 0.98, f"PC{xi+1}: {ve[xi]*100:.0f}%  PC{yi+1}: {ve[yi]*100:.0f}%\n"
                        f"var. explained (top {len(ve)} PCs)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            style="italic", color="0.3")
    ax.set_xlabel(f"PC{xi+1}"); ax.set_ylabel(f"PC{yi+1}")
    handles = [plt.Line2D([0], [0], color="0.2", ls="-", lw=1.8, label="pre-reversal"),
              plt.Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="post-reversal")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"{F.MODELS[model_type]['label']} PCA phase space\n"
                 f"(n={D['n_pre']} pre, {D['n_post']}/{D['n_post_total']} recovered post)",
                 fontsize=10)
    return fig


def draw_pca_timecourses(model_type, n_components=3):
    D = PPCA.population_pca_trajectories(model_type, n_components=n_components)
    t = D["t"]
    n_iti, stim_ts = D["period"]["n_iti_pre"], D["period"]["stim_ts"]
    k = D["pre_mean"].shape[-1]
    fig, axes = plt.subplots(k, 1, figsize=(6.2, 2.6 * k), sharex=True)
    axes = np.atleast_1d(axes)
    ve = D["var_explained"]
    for pc, ax in enumerate(axes):
        ax.axvline(-0.5, color="0.6", lw=0.8, ls=":")
        ax.axvline(stim_ts - 0.5, color="0.6", lw=0.8, ls=":")
        for si, stim in enumerate(STIM_ORDER):
            colour = S.STIM_COLOURS[stim]
            pm, ps = D["pre_mean"][:, si, pc], D["pre_sem"][:, si, pc]
            qm, qs = D["post_mean"][:, si, pc], D["post_sem"][:, si, pc]
            ax.fill_between(t, pm - ps, pm + ps, color=colour, alpha=0.15, lw=0)
            ax.plot(t, pm, color=colour, ls="-", lw=1.8,
                    label=f"{STIM_PRETTY[stim]}, pre" if pc == 0 else None)
            if np.isfinite(qm).all():
                ax.fill_between(t, qm - qs, qm + qs, color=colour, alpha=0.15, lw=0)
                ax.plot(t, qm, color=colour, ls="--", lw=1.8,
                        label=f"{STIM_PRETTY[stim]}, post" if pc == 0 else None)
        ax.set_ylabel(f"PC{pc+1}\n({ve[pc]*100:.0f}% var.)")
        ax.set_xlim(t[0], t[-1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time from stim onset (bins)")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left",
                   bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{F.MODELS[model_type]['label']} PCA component time courses "
                 f"(n={D['n_pre']} pre, {D['n_post']}/{D['n_post_total']} recovered post)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def draw_pca_phase_space_grid(model_type, pcs=(0, 1), ncols=6):
    """Supplementary (record of every seed, per the chat): one panel per
    seed, its own per-seed joint-basis PCA phase-space trajectory (pre
    solid; post dashed only if this seed is in the recovered-seeds set --
    see population_pca.py's module docstring for the seed-inclusion
    convention), same start/end marker convention as draw_pca_phase_space."""
    D = PPCA.population_pca_per_seed_trajectories(model_type)
    t = D["t"]
    m = t >= 0
    xi, yi = pcs
    seeds = D["seeds"]
    n = len(seeds)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False)
    for i, d in enumerate(seeds):
        ax = axes[i // ncols][i % ncols]
        for si, stim in enumerate(STIM_ORDER):
            colour = S.STIM_COLOURS[stim]
            x, y = d["scores_pre"][m, si, xi], d["scores_pre"][m, si, yi]
            ax.plot(x, y, color=colour, ls="-", lw=1.0)
            ax.scatter(x[-1], y[-1], color=colour, s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
            if d["has_post"]:
                x2, y2 = d["scores_post"][m, si, xi], d["scores_post"][m, si, yi]
                ax.plot(x2, y2, color=colour, ls="--", lw=1.0)
                ax.scatter(x2[-1], y2[-1], color=colour, s=18, marker="s", edgecolor="k", lw=0.3, zorder=3)
        ax.set_title(f"seed {d['seed']}" + ("" if d["has_post"] else " (no post)"), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{F.MODELS[model_type]['label']} PCA phase space (PC{xi + 1}-PC{yi + 1}) "
                f"-- every seed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def draw_pca_phase_space_3d(model_type, pcs=(0, 1, 2), ax=None, elev=22, azim=-60):
    """3D analogue of draw_pca_phase_space -- same trajectories (one line
    per stimulus, solid pre-reversal / dashed post-reversal, same
    start/end marker convention), but through PC1-PC2-PC3 jointly instead
    of a 2D PC1-PC2 projection, on request ("so you can see PC3 as well").
    Requires mpl_toolkits.mplot3d (stdlib-adjacent, ships with matplotlib).
    If ax is given it must already be a 3D axes (projection="3d") -- e.g.
    fig.add_subplot(gs[r, c], projection="3d"); when ax is None (the normal
    standalone-figure case) one is created here."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    if ax is None:
        fig = plt.figure(figsize=(8.6, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.02, right=0.78, top=0.95, bottom=0.08)
    else:
        fig = ax.figure
    D = PPCA.population_pca_trajectories(model_type, n_components=max(pcs) + 1)
    t = D["t"]
    m = t >= 0
    xi, yi, zi = pcs
    for si, stim in enumerate(STIM_ORDER):
        colour = S.STIM_COLOURS[stim]
        for mean, ls, lw in ((D["pre_mean"], "-", 1.8), (D["post_mean"], "--", 1.8)):
            x, y, z = mean[m, si, xi], mean[m, si, yi], mean[m, si, zi]
            ax.plot(x, y, z, color=colour, ls=ls, lw=lw)
            ax.scatter(x[-1], y[-1], z[-1], color=colour, s=32, marker="s",
                       edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(D["pre_mean"][m, si, xi][0], D["pre_mean"][m, si, yi][0],
                   D["pre_mean"][m, si, zi][0], color=colour, s=24, zorder=3)
    ve = D["var_explained"]
    ax.set_xlabel(f"PC{xi+1} ({ve[xi]*100:.0f}%)")
    ax.set_ylabel(f"PC{yi+1} ({ve[yi]*100:.0f}%)")
    ax.tick_params(axis="both", pad=2)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    # ax.set_zlabel() places its label at a hardcoded, view-angle-independent
    # y=0 display pixel in this matplotlib version (a known mplot3d bug --
    # verified empirically across every elev/azim combination), which
    # renders it clipped off the bottom of the canvas every time. Placed
    # manually instead, in axes-fraction coordinates along the right edge,
    # which is immune to that bug.
    ax.text2D(1.05, 0.5, f"PC{zi+1} ({ve[zi]*100:.0f}%)", transform=ax.transAxes,
              rotation=90, ha="center", va="center", fontsize=13)
    ax.view_init(elev=elev, azim=azim)
    handles = [plt.Line2D([0], [0], color="0.2", ls="-", lw=1.8, label="pre-reversal"),
              plt.Line2D([0], [0], color="0.2", ls="--", lw=1.8, label="post-reversal")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{F.MODELS[model_type]['label']} PCA phase space (3D)\n"
                 f"(n={D['n_pre']} pre, {D['n_post']}/{D['n_post_total']} recovered post)",
                 fontsize=10)
    return fig


def build_all(show_tag=None):
    for mt in MODEL_TYPES:
        try:
            save_panel(draw_pca_phase_space_grid(mt), "Dimensionality/PCA",
                       f"DIM.PCA.grid.{mt}", f"{mt}_pca_phase_grid_all_seeds", show_tag)
        except Exception as e:
            print(f"  (skip PCA seed grid for {mt}: {e})")
        try:
            save_panel(draw_pca_phase_space(mt), "Dimensionality/PCA",
                       f"DIM.PCA.phase.{mt}", f"{mt}_pca_phase_space", show_tag)
        except Exception as e:
            print(f"  (skip PCA phase space for {mt}: {e})")
        try:
            save_panel(draw_pca_phase_space_3d(mt), "Dimensionality/PCA",
                       f"DIM.PCA.phase3d.{mt}", f"{mt}_pca_phase_space_3d", show_tag)
        except Exception as e:
            print(f"  (skip PCA phase space 3D for {mt}: {e})")
        try:
            save_panel(draw_pca_timecourses(mt), "Dimensionality/PCA",
                       f"DIM.PCA.tc.{mt}", f"{mt}_pca_timecourses", show_tag)
        except Exception as e:
            print(f"  (skip PCA timecourses for {mt}: {e})")


if __name__ == "__main__":
    build_all()
