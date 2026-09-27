"""Behaviour panels (anticipatory lick rate).

Tags:
  BEH.EXP.lickbar      Expert sessions lick bar plot
  BEH.EXP.licktrials   Expert sessions trial-resolved lick rates (all sessions)
  BEH.REV.licktrials   Reversal sessions trial-resolved lick rates (all sessions)
  BEH.REV.lickcombined Reversal sessions combined, reversal-aligned (per-stimulus
                       20-trial rolling mean, mean ± SEM over sessions)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _tags import DATA, new_panel, save_panel

try:
    from sat_plot_colours import STIM_STEM_COLOURS
except Exception:
    STIM_STEM_COLOURS = {"100_to_0": "#4b2362", "50": "#c24167", "0_to_100": "#edb081"}

# ---- stimulus colour / label schemes ---------------------------------------
REV_ORDER = ["100_to_0", "50", "0_to_100"]
REV_PRETTY = {"100_to_0": "100%→0%", "50": "50%", "0_to_100": "0%→100%"}
EXP_ORDER = ["100", "50", "0"]
EXP_PRETTY = {"100": "100%", "50": "50%", "0": "0%"}
EXP_COLOURS = {"100": STIM_STEM_COLOURS["100_to_0"], "50": STIM_STEM_COLOURS["50"],
               "0": STIM_STEM_COLOURS["0_to_100"]}
ROLL = 15
COMBINED_BIN = 20   # reversal combined smoothing/bin width (trials)

REV_CSV = DATA / "lick" / "lick_rates_per_trial.csv"
EXP_CSV = DATA / "lick" / "lick_rates_per_trial_expert.csv"
META = DATA / "single_trial" / "single_trial_metadata.csv"
DECODE_CSV = DATA / "decoding" / "decoding_timepooled_per_session.csv"
# REVISITED (see chat): raising this to 0.70 to "cleanly separate" decodable
# sessions was the wrong fix -- it dropped n from 7 to 6 qualifying sessions
# and broke the plot further (two of the three stimulus traces had no trial
# range left with >=min_sessions coverage). Checked empirically instead:
# rendering the panel at threshold=0.55 (11 sessions), and with no filter at
# all (13 sessions) produce NEARLY IDENTICAL traces/SEM to each other and are
# both far cleaner than any higher threshold -- i.e. under the new (more
# conservative, 100-neuron-cap) decoder, stim_post_0v1 accuracy just doesn't
# separate sessions into behaviourally-different groups any more, so filtering
# by it only shrinks n without curating anything. Default is now unfiltered;
# pass an explicit decode_threshold= to re-enable filtering if ever wanted.
DECODE_THRESHOLD = None  # min post-reversal 0-vs-100% stim decoding accuracy (None = no filter)


def _reversal_points():
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


def _qualifying_sessions(threshold=DECODE_THRESHOLD):
    """Reversal sessions passing a minimum post-reversal 0-vs-100% stimulus
    decoding accuracy (decoder='stim_post_0v1', scope='reversal'). Returns
    None (= no filtering, use every session) if the decoding CSV isn't
    available or has no matching rows."""
    if not DECODE_CSV.exists():
        return None
    d = pd.read_csv(DECODE_CSV)
    sub = d[(d.scope == "reversal") & (d.decoder == "stim_post_0v1")]
    if sub.empty:
        return None
    return set(sub[sub.accuracy > threshold].session)


def _has(csv):
    return csv.exists() and len(pd.read_csv(csv)) > 0


# ---- expert lick bar (BEH.EXP.lickbar) -------------------------------------
def draw_expert_lick_bar(ax=None):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    df = pd.read_csv(EXP_CSV)
    per = df.groupby(["stimulus_label", "session"]).anticipatory_lick_rate.mean().reset_index()
    order = [l for l in EXP_ORDER if l in per.stimulus_label.astype(str).unique()]
    rng = np.random.default_rng(0)
    for x, lab in enumerate(order):
        vals = per[per.stimulus_label.astype(str) == lab].anticipatory_lick_rate.values
        m = np.nanmean(vals)
        se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.bar(x, m, width=0.65, color=EXP_COLOURS[lab],
               yerr=se, capsize=4, zorder=2)
        ax.scatter(x + (rng.random(len(vals)) - 0.5) * 0.25, vals, color="0.3",
                   s=13, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([EXP_PRETTY[l] for l in order])
    ax.set_xlabel("stimulus (rew. prob.)")
    ax.set_ylabel("lick rate (Hz)")
    ax.set_title(f"Expert lick rate by stimulus (n={per.session.nunique()} sessions)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---- trial-resolved small-multiples (expert + reversal) --------------------
def _draw_trials_grid(csv, order, pretty, colours, draw_reversal, title):
    df = pd.read_csv(csv)
    revs = _reversal_points() if draw_reversal else {}
    sessions = sorted(df.session.unique())
    ncols = 4
    nrows = -(-len(sessions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                             squeeze=False)
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sdf = df[df.session == sess]
        for lab in order:
            s = sdf[sdf.stimulus_label.astype(str) == lab].sort_values("trial_number")
            if s.empty:
                continue
            y = s.anticipatory_lick_rate.rolling(ROLL, min_periods=1, center=True).mean()
            ax.plot(s.trial_number, y, color=colours[lab], lw=1.5, label=pretty[lab])
        if sess in revs:
            ax.axvline(revs[sess], color="0.35", ls="--", lw=1.1)
        ax.set_title(sess, fontsize=8)
        ax.set_xlabel("trial", fontsize=8); ax.set_ylabel("Hz", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(len(sessions), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def draw_expert_lick_trials(ax=None):
    return _draw_trials_grid(EXP_CSV, EXP_ORDER, EXP_PRETTY, EXP_COLOURS, False,
                             f"Expert anticipatory lick rate over trials ({ROLL}-trial mean)")


def draw_reversal_lick_trials(ax=None):
    return _draw_trials_grid(REV_CSV, REV_ORDER, REV_PRETTY, STIM_STEM_COLOURS, True,
                             f"Reversal anticipatory lick rate over trials "
                             f"(dashed=reversal; {ROLL}-trial mean)")


# ---- reversal combined, reversal-aligned (BEH.REV.lickcombined) ------------
def draw_reversal_lick_combined(ax=None, roll=COMBINED_BIN, min_sessions=5,
                                sessions=None, decode_threshold=DECODE_THRESHOLD):
    """Lick rate vs trial relative to each session's reversal, mean ± SEM over
    sessions, per stimulus.

    Method: raw per-trial values are first aligned across sessions at their
    actual trial-relative-to-reversal position (rounded to the nearest
    integer trial; no interpolation/upsampling -- a session only contributes
    at a trial index it actually has), averaged across sessions, and *then*
    smoothed with a centred ``roll``-trial rolling mean (applied to both the
    mean and SEM traces). A point is only shown where at least
    ``min_sessions`` sessions contribute at that trial index.

    Sessions default to those passing a post-reversal 0-vs-100% stimulus
    decoding accuracy threshold (see ``_qualifying_sessions``); pass an
    explicit ``sessions`` to override, or ``decode_threshold=None`` to use
    every session with a reversal point.

    The x-range shown is cropped to the *intersection*, across all 3
    stimuli, of where each stimulus individually has >=min_sessions data --
    so we never show a stretch of trials for one stimulus that isn't backed
    by data for the other two.

    Args:
        roll: rolling-mean window in trials, applied after averaging.
        min_sessions: min sessions contributing to a trial index for it to show.
        sessions: optional explicit iterable of session names to include.
            None (default) = auto-select via decode_threshold.
        decode_threshold: min post-reversal 0v1 stim decoding accuracy for a
            session to be auto-included. Ignored if `sessions` is given.
    """
    fig, ax, _ = new_panel(ax, figsize=(8, 5))
    df = pd.read_csv(REV_CSV)
    revs = _reversal_points()
    df = df[df.session.isin(revs)].copy()
    if sessions is None and decode_threshold is not None:
        sessions = _qualifying_sessions(decode_threshold)
    if sessions is not None:
        keep = set(sessions)
        df = df[df.session.isin(keep)].copy()
        if df.empty:
            raise ValueError("No matching sessions after filtering.")
    df["rel"] = df.trial_number - df.session.map(revs)

    per_label = {}
    for lab in REV_ORDER:
        cols = {}                                     # session -> Series(index=trial, raw)
        for sess, sdf in df[df.stimulus_label.astype(str) == lab].groupby("session"):
            s = sdf.sort_values("trial_number")
            if len(s) < 2:
                continue
            ri = np.rint(s.rel.values).astype(int)    # actual trial index (rel), integer
            # if two of this session's trials round to the same index, average them
            cols[sess] = pd.Series(s.anticipatory_lick_rate.values, index=ri).groupby(level=0).mean()
        if not cols:
            per_label[lab] = None
            continue
        # align sessions on the union of their actual trial indices (NaN = no trial)
        M = pd.DataFrame(cols).sort_index()
        cnt = M.notna().sum(axis=1)
        mean = M.mean(axis=1)
        sem = M.std(axis=1, ddof=1) / np.sqrt(cnt.clip(lower=1))
        keep_mask = cnt >= min_sessions
        gx = M.index.values[keep_mask.values]
        mean_k = pd.Series(mean.values[keep_mask.values], index=gx)
        sem_k = pd.Series(sem.values[keep_mask.values], index=gx)
        # smooth AFTER averaging across sessions (mean and SEM traces both)
        mean_s = mean_k.rolling(roll, min_periods=1, center=True).mean()
        sem_s = sem_k.rolling(roll, min_periods=1, center=True).mean()
        per_label[lab] = (gx, mean_s.values, sem_s.values)

    # crop the trial-range to where ALL 3 stimuli have data
    have_all = all(per_label[lab] is not None for lab in REV_ORDER)
    lo = hi = None
    if have_all:
        lo = max(per_label[lab][0].min() for lab in REV_ORDER)
        hi = min(per_label[lab][0].max() for lab in REV_ORDER)

    for lab in REV_ORDER:
        v = per_label[lab]
        if v is None:
            continue
        gx, gm, gs = v
        if lo is not None:
            m = (gx >= lo) & (gx <= hi)
            gx, gm, gs = gx[m], gm[m], gs[m]
        ax.plot(gx, gm, color=STIM_STEM_COLOURS[lab], lw=2.2, label=REV_PRETTY[lab])
        ax.fill_between(gx, gm - gs, gm + gs, color=STIM_STEM_COLOURS[lab], alpha=0.2, lw=0)
    ax.axvline(0, color="0.35", ls="--", lw=1.4, label="reversal")
    if lo is not None:
        ax.set_xlim(lo, hi)   # exact crop -- no default matplotlib margin past the data
    ax.set_xlabel("trials from reversal")
    ax.set_ylabel("lick rate (Hz)")
    n_sess = df.session.nunique()
    crop_txt = f", trials [{int(lo)}, {int(hi)}]" if lo is not None else ""
    ax.set_title(f"Reversal lick rate, session-combined (avg-then-{roll}-trial-smooth "
                 f"per stimulus,\nmean ± SEM over {n_sess} sessions{crop_txt})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    return fig


def build_all(show_tag=None):
    if _has(EXP_CSV):
        save_panel(draw_expert_lick_bar(), "Behaviour/Expert", "BEH.EXP.lickbar",
                   "expert_lick_bar", show_tag)
        save_panel(draw_expert_lick_trials(), "Behaviour/Expert", "BEH.EXP.licktrials",
                   "expert_lick_trials", show_tag)
    else:
        print("  (expert lick CSV empty — skipping BEH.EXP.*)")
    if _has(REV_CSV):
        save_panel(draw_reversal_lick_trials(), "Behaviour/Reversal", "BEH.REV.licktrials",
                   "reversal_lick_trials", show_tag)
        save_panel(draw_reversal_lick_combined(), "Behaviour/Reversal", "BEH.REV.lickcombined",
                   "reversal_lick_combined", show_tag)


if __name__ == "__main__":
    build_all()
