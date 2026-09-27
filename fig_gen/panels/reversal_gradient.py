"""Post-reversal learning-rate (gradient) bar panel -- real mouse behaviour.

Summarizes the reversal-aligned lick-rate curve (see behaviour.py's
BEH.REV.lickcombined / draw_reversal_lick_combined) as ONE number per
stimulus direction instead of the full trial-resolved curve: the ordinary-
least-squares slope of per-trial anticipatory lick rate vs. trials-since-
reversal, computed per SESSION over a fixed window of trials right after
the reversal (default: the first 100 trials; a 200-trial variant is also
provided), then shown as mean +/- SEM across sessions with each session's
own slope plotted as a jittered point.

Only the two VALUE-REVERSING stimulus directions (100%->0%, 0%->100%) are
shown -- the 50% stimulus is an unchanged control, not a "gradient" question.

A fixed window (rather than a per-session "recovery point", which is how
the RNN side's analogous panel is defined -- see combined/analysis/
reversal_gradients.py -- and see the chat for why the two data sources use
different definitions) is used here because real behavioural sessions are
short, noisy, and don't reach a clean stable post-reversal asymptote the
way a densely-probed RNN training run does; a fixed early window is the
more robust, less assumption-laden choice for this side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _tags import DATA, new_panel, save_panel

try:
    from sat_plot_colours import STIM_STEM_COLOURS
except Exception:
    STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}

REV_CSV = DATA / "lick" / "lick_rates_per_trial.csv"
META = DATA / "single_trial" / "single_trial_metadata.csv"

# 0%->100% first, then 100%->0% -- consistent order used across every new
# gradient panel (this file and combined/panels/reversal_gradient_bars.py),
# so the RNN and experimental panels line up column-for-column once combined.
GRADIENT_STIMS = ("0_to_100", "100_to_0")
GRADIENT_PRETTY = {"100_to_0": "100%→0%", "0_to_100": "0%→100%"}


def _reversal_points():
    """{session: reversal boundary in session_trial_number units} -- see
    behaviour.py's own _reversal_points() (identical logic, duplicated here
    so this panel has no import-order dependency on behaviour.py)."""
    if not META.exists():
        return {}
    m = pd.read_csv(META)
    out = {}
    for sess, g in m.groupby("session"):
        pre = g[g.condition.str.contains("pre-Rev")].session_trial_number
        post = g[g.condition.str.contains("post-Rev")].session_trial_number
        if len(pre) and len(post):
            out[sess] = 0.5 * (pre.max() + post.min())
    return out


def fixed_window_gradient(window=100, stims=GRADIENT_STIMS, min_trials=5):
    """DataFrame [session, stimulus, n_trials, slope] -- one row per
    (session, stimulus) with >= min_trials real per-trial lick-rate points
    inside [0, window) trials-since-reversal. slope is Hz gained per trial
    (OLS fit over just that window) -- positive means lick rate climbing
    toward the new post-reversal target."""
    if not REV_CSV.exists():
        return pd.DataFrame(columns=["session", "stimulus", "n_trials", "slope"])
    df = pd.read_csv(REV_CSV)
    revs = _reversal_points()
    df = df[df.session.isin(revs)].copy()
    df["rel"] = df.trial_number - df.session.map(revs)
    rows = []
    for sess, sdf in df.groupby("session"):
        for stim in stims:
            s = sdf[(sdf.stimulus_label.astype(str) == stim) & (sdf.rel >= 0) & (sdf.rel < window)]
            if len(s) < min_trials:
                continue
            slope = float(np.polyfit(s.rel.values, s.anticipatory_lick_rate.values, 1)[0])
            rows.append({"session": sess, "stimulus": stim, "n_trials": len(s), "slope": slope})
    return pd.DataFrame(rows)


def session_window_points(window=100, stims=GRADIENT_STIMS, min_trials=5):
    """Like fixed_window_gradient(), but keeps the raw (rel, lick_rate)
    points per (session, stimulus) instead of collapsing straight to the
    slope -- for draw_fit_diagnostic_grid() below, which needs to actually
    SHOW the points each session's OLS line was fit to, not just report the
    resulting number. Returns {(session, stimulus): {x, y, slope,
    intercept}}."""
    if not REV_CSV.exists():
        return {}
    df = pd.read_csv(REV_CSV)
    revs = _reversal_points()
    df = df[df.session.isin(revs)].copy()
    df["rel"] = df.trial_number - df.session.map(revs)
    out = {}
    for sess, sdf in df.groupby("session"):
        for stim in stims:
            s = sdf[(sdf.stimulus_label.astype(str) == stim) & (sdf.rel >= 0) & (sdf.rel < window)]
            if len(s) < min_trials:
                continue
            x = s.rel.values.astype(float)
            y = s.anticipatory_lick_rate.values.astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            out[(sess, stim)] = dict(x=x, y=y, slope=float(slope), intercept=float(intercept))
    return out


def draw_fit_diagnostic_grid(window=100, ncols=5):
    """Supplementary grid (record of every session, per the chat): one panel
    per reversal session, the raw per-trial anticipatory lick-rate points
    actually used by fixed_window_gradient()'s OLS fit (within [0, window)
    trials-since-reversal), one colour per value-reversing stimulus
    direction, fit line overlaid on top of its own points -- lets you check
    the fit itself, not just the slope number it produces."""
    pts = session_window_points(window=window)
    sessions = sorted({sess for sess, _ in pts})
    n = len(sessions)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        for stim in GRADIENT_STIMS:
            d = pts.get((sess, stim))
            if d is None:
                continue
            colour = STIM_STEM_COLOURS[stim]
            ax.scatter(d["x"], d["y"], s=10, color=colour, alpha=0.6, linewidth=0)
            xx = np.array([0, window])
            ax.plot(xx, d["intercept"] + d["slope"] * xx, color=colour, lw=1.8)
        ax.set_title(sess.split("_")[-1] + " " + sess[:10], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Reversal lick-rate gradient fits -- every session (first {window} trials)",
                fontsize=11)
    fig.text(0.5, 0.01, "trials since reversal", ha="center", fontsize=9)
    fig.text(0.01, 0.5, "anticipatory lick rate (Hz)", va="center", rotation=90, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    return fig


COMBINED_ROLL = 20   # same COMBINED_BIN default as behaviour.py's draw_reversal_lick_combined


def cross_session_average_window_points(window=100, stims=GRADIENT_STIMS, min_sessions=5,
                                        roll=COMBINED_ROLL):
    """Cross-session-averaged counterpart of session_window_points(): the
    line for draw_cross_session_fit() below is fit to ONE aggregate curve
    per stimulus, not to each session's own data -- built with EXACTLY the
    same cross-session averaging, session-inclusion floor, and post-average
    smoothing already established for the reversal lick-rate curve panel
    (behaviour.py's draw_reversal_lick_combined): each session's per-trial
    values are aligned at their own integer trials-since-reversal index (no
    interpolation -- a session only contributes at an index it actually
    has), averaged across sessions at each index, an index is only kept if
    >= `min_sessions` sessions contribute there (same min_sessions=5
    default), and the resulting mean is smoothed AFTER averaging with a
    centred `roll`-trial rolling mean (same roll=20 / COMBINED_BIN default).
    Restricted here to [0, window) trials-since-reversal and to the two
    gradient-relevant stimulus directions (see module docstring).

    Returns {stimulus: {x, y, slope, intercept, n_sessions_min}} --
    n_sessions_min is the fewest sessions contributing to any single point
    in the window (for reporting alongside the fit)."""
    if not REV_CSV.exists():
        return {}
    df = pd.read_csv(REV_CSV)
    revs = _reversal_points()
    df = df[df.session.isin(revs)].copy()
    df["rel"] = df.trial_number - df.session.map(revs)
    out = {}
    for stim in stims:
        cols = {}
        sdf_all = df[(df.stimulus_label.astype(str) == stim) & (df.rel >= 0) & (df.rel < window)]
        for sess, sdf in sdf_all.groupby("session"):
            s = sdf.sort_values("trial_number")
            if len(s) < 2:
                continue
            ri = np.rint(s.rel.values).astype(int)
            cols[sess] = pd.Series(s.anticipatory_lick_rate.values, index=ri).groupby(level=0).mean()
        if not cols:
            continue
        M = pd.DataFrame(cols).sort_index()
        cnt = M.notna().sum(axis=1)
        mean = M.mean(axis=1)
        keep_mask = (cnt >= min_sessions).values
        if not keep_mask.any():
            continue
        gx = M.index.values[keep_mask]
        mean_k = pd.Series(mean.values[keep_mask], index=gx)
        mean_s = mean_k.rolling(roll, min_periods=1, center=True).mean()
        x = gx.astype(float)
        y = mean_s.values
        slope, intercept = np.polyfit(x, y, 1)
        out[stim] = dict(x=x, y=y, slope=float(slope), intercept=float(intercept),
                          n_sessions_min=int(cnt.values[keep_mask].min()))
    return out


def draw_cross_session_fit(window=100, min_sessions=5, roll=COMBINED_ROLL, ax=None):
    """The cross-session-average counterpart of draw_fit_diagnostic_grid()
    (per the chat): instead of one panel per session, ONE panel with the
    cross-session-averaged lick-rate curve per stimulus direction (points,
    same inclusion criterion/filtering/smoothing as behaviour.py's
    draw_reversal_lick_combined -- see cross_session_average_window_points())
    and the OLS line fit to THAT averaged curve overlaid -- i.e. what the
    gradient looks like if you fit to the group-average trace instead of
    averaging per-session slopes (compare against draw_reversal_gradient_
    bars(), which does the latter)."""
    fig, ax, _ = new_panel(ax, figsize=(5.2, 4.8))
    fits = cross_session_average_window_points(window=window, min_sessions=min_sessions, roll=roll)
    for stim in GRADIENT_STIMS:
        d = fits.get(stim)
        if d is None:
            continue
        colour = STIM_STEM_COLOURS[stim]
        ax.plot(d["x"], d["y"], color=colour, lw=1.2, alpha=0.85,
                label=f"{GRADIENT_PRETTY[stim]} (>= {d['n_sessions_min']} sessions/pt)")
        xx = np.array([d["x"][0], d["x"][-1]])
        ax.plot(xx, d["intercept"] + d["slope"] * xx, color=colour, lw=2.4, ls="--")
    ax.set_xlabel("trials since reversal")
    ax.set_ylabel("anticipatory lick rate (Hz), cross-session mean")
    ax.set_title(f"Reversal lick-rate gradient fit to the cross-session average\n"
                f"(first {window} trials, >= {min_sessions} sessions/point, "
                f"{roll}-trial rolling mean)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _draw_gradient_bars(ax, items, title, ylabel, rng_seed=0):
    """items: [(label, colour, values_array, count_text)], one bar each."""
    rng = np.random.default_rng(rng_seed)
    for i, (label, colour, vals, count_text) in enumerate(items):
        if len(vals):
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.bar(i, m, width=0.6, color=colour, yerr=se, capsize=4, zorder=2)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(i + jitter, vals, color="0.25", s=15, alpha=0.6, zorder=3, linewidth=0)
    ax.axhline(0, color="0.5", lw=1, zorder=1)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{lab}\n({ct})" for lab, _, _, ct in items], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def draw_reversal_gradient_bars(window=100, ax=None):
    """The BEH.REV panel-f replacement: bar plot of the fixed-`window`-trial
    lick-rate gradient, one bar per value-reversing stimulus direction, mean
    +/- SEM across sessions with each session's own slope jittered on top."""
    fig, ax, _ = new_panel(ax, figsize=(4.6, 4.6))
    df = fixed_window_gradient(window=window)
    items = []
    for stim in GRADIENT_STIMS:
        vals = df[df.stimulus == stim].slope.values
        items.append((GRADIENT_PRETTY[stim], STIM_STEM_COLOURS[stim], vals,
                      f"n={len(vals)} sessions"))
    _draw_gradient_bars(ax, items,
                        f"Reversal lick-rate gradient\n(first {window} trials post-reversal)",
                        "slope (Hz / trial)")
    return fig


def build_all(show_tag=None):
    for window in (100, 200):
        save_panel(draw_reversal_gradient_bars(window=window), "Behaviour/Reversal",
                   f"BEH.REV.gradient{window}", f"reversal_gradient_bars_{window}", show_tag)
        save_panel(draw_fit_diagnostic_grid(window=window), "Behaviour/Reversal",
                   f"BEH.REV.gradient{window}.fitdiag", f"reversal_gradient_fit_diagnostic_{window}",
                   show_tag)
        save_panel(draw_cross_session_fit(window=window), "Behaviour/Reversal",
                   f"BEH.REV.gradient{window}.crosssession", f"reversal_gradient_cross_session_fit_{window}",
                   show_tag)


if __name__ == "__main__":
    build_all()
