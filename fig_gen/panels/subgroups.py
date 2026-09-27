"""Subgroup panels — significant responders, responsiveness heatmaps, Sankey.

Responders are defined by the temporal t-test criterion (see
extract/extract_responsiveness_ttest.py + analysis/responsiveness_ttest.py):
per neuron, per stimulus, an independent t-test at each timepoint compares the
stimulus window (0-2 s) against the pre-stimulus baseline; the neuron is a
significant responder to that stimulus when its longest contiguous run of
significant timepoints exceeds n_timepoints/3. Significance is **per stimulus**
(a neuron can respond to more than one), so the size bars double-count.

Reads data/subgroups/responsiveness_ttest_{reversal,expert}_long.csv
  reversal: session, neuron_id, phase, stimulus, sig, effect, resp
  expert:   session, neuron_id, stimulus, sig, effect, resp
where `resp` is the avg stimulus response (trial-averaged mean dF/F in the 0-2 s
stimulus window, baseline-subtracted) — this is what the heatmaps display, and a
cell's preferred stimulus = the condition with the largest `resp`.

Tags:
  SUB.EXP.sizes         Expert per-stimulus responder counts bar
  SUB.EXP.sizesfine     Expert selectivity-group bar (pure + mixed sets, e.g. 0%&50%)
  SUB.EXP.heatmap       Expert neuron x stimulus responsiveness heatmaps (per session)
  SUB.EXP.heatmappool   Expert neuron x stimulus responsiveness heatmap (all cells pooled)
  SUB.REV.sizes         Reversal per-stimulus responder counts bar (pre & post)
  SUB.REV.sizesfine     Reversal selectivity-group bar (pure + mixed sets, pre & post)
  SUB.REV.heatmap       Reversal neuron x condition responsiveness heatmaps (per session)
  SUB.REV.heatmappool   Reversal neuron x condition responsiveness heatmap (all cells pooled)
  SUB.REV.sankey        Reversal responder pre->post Sankey, per session
  SUB.REV.sankeycomb    Reversal responder pre->post Sankey, sessions combined
  SUB.REV.sankeyfine    Reversal selectivity-group pre->post Sankey (fine sets + non-resp)
  SUB.REV.sankeyfineresp  Reversal selectivity-group pre->post Sankey (responders only)

Fine-grained "selectivity groups" classify each cell by the *set* of stimuli it
significantly responds to: pure single-stimulus groups, mixed pairs (e.g. 0% &
50%), the all-three group, and non-responsive. Standalone panels (not in Fig 1).
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

SUB = DATA / "subgroups"
HOT = "hot"
NONRESP = "non-responsive"
REV_ORDER = ["100_to_0", "50", "0_to_100"]
EXP_ORDER = ["100", "50", "0"]
GROUP_COL = {"100_to_0": STIM_STEM_COLOURS["100_to_0"], "50": STIM_STEM_COLOURS["50"],
             "0_to_100": STIM_STEM_COLOURS["0_to_100"],
             "100": STIM_STEM_COLOURS["100_to_0"], "0": STIM_STEM_COLOURS["0_to_100"],
             NONRESP: "0.7"}
PRETTY = {"100_to_0": "100%→0%", "0_to_100": "0%→100%", "50": "50%",
          "100": "100%", "0": "0%", NONRESP: "non-resp"}
REV_COND_ORDER = ["100_to_0", "50", "0_to_100"]   # per phase

# --- Responder-computation method selector -----------------------------
# "temporal" (default): responsiveness_ttest.py's per-timepoint contiguous-
#   significant-run criterion (the original neural-side method).
# "time_averaged": responsiveness_time_averaged.py's window-mean one-sample
#   t-test (mirrors the model side's own existing method), so the two data
#   types can be compared method-for-method (see extract/
#   extract_responsiveness_time_averaged.py and the chat).
# Set as a module attribute -- mirrors this bundle's REV_TAG convention
# elsewhere (env-var-style global read at call time) rather than threading a
# `method=` parameter through every draw_* function here; compose.py sets
# subgroups.METHOD before building a method-specific figure variant.
METHOD = "temporal"
_METHOD_FILES = {
    "temporal": ("responsiveness_ttest_reversal_long.csv",
                "responsiveness_ttest_expert_long.csv"),
    "time_averaged": ("responsiveness_time_averaged_reversal_long.csv",
                      "responsiveness_time_averaged_expert_long.csv"),
}


def _rev_long():
    return _METHOD_FILES[METHOD][0]


def _exp_long():
    return _METHOD_FILES[METHOD][1]


def _has(f):
    return (SUB / f).exists()


def _n_cells(df):
    return int(df[["session", "neuron_id"]].drop_duplicates().shape[0])


def _any_sig(sdf, mat):
    """Bool array, one per row of ``mat`` (aligned to mat.index): True if the
    cell is a significant responder to >=1 of the displayed conditions (any
    stimulus, any phase) -- used to separate non-responders in the heatmaps."""
    any_sig = sdf.groupby(["session", "neuron_id"]).sig.any()
    return any_sig.reindex(mat.index).fillna(False).to_numpy(dtype=bool)


# --------------------------------------------------------------------------- #
# Responder group sizes (per stimulus; a neuron may count in several bars)
# --------------------------------------------------------------------------- #
def draw_expert_sizes(ax=None, include_nonresp=True):
    fig, ax, _ = new_panel(ax, figsize=(5, 4.5))
    df = pd.read_csv(SUB / _exp_long())
    df["stimulus"] = df["stimulus"].astype(str)
    cats = EXP_ORDER + ([NONRESP] if include_nonresp else [])
    counts = [int(((df.stimulus == c) & df.sig).sum()) for c in EXP_ORDER]
    if include_nonresp:
        any_sig = df.groupby(["session", "neuron_id"]).sig.any().sum()
        counts.append(int(_n_cells(df) - any_sig))
    ax.bar(range(len(cats)), counts, color=[GROUP_COL[c] for c in cats])
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([PRETTY[c] for c in cats])
    ax.set_ylabel("number of cells"); ax.set_xlabel("responsive to stimulus")
    ax.set_title(f"Expert significant responders ({df.session.nunique()} sessions, "
                 f"{_n_cells(df)} neurons)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_reversal_sizes(ax=None, include_nonresp=True):
    fig, ax, _ = new_panel(ax, figsize=(6.5, 4.5))
    df = pd.read_csv(SUB / _rev_long())
    df["stimulus"] = df["stimulus"].astype(str); df["phase"] = df["phase"].astype(str)
    cats = REV_ORDER + ([NONRESP] if include_nonresp else [])
    w = 0.38
    for phase in ["pre", "post"]:
        p = df[df.phase == phase]
        counts = [int(((p.stimulus == c) & p.sig).sum()) for c in REV_ORDER]
        if include_nonresp:
            any_sig = p.groupby(["session", "neuron_id"]).sig.any().sum()
            counts.append(int(_n_cells(p) - any_sig))
        off = -w / 2 if phase == "pre" else w / 2
        ax.bar(np.arange(len(cats)) + off, counts, width=w,
               color=[GROUP_COL[c] for c in cats],
               alpha=1.0 if phase == "pre" else 0.55,
               label="pre" if phase == "pre" else "post")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([PRETTY[c] for c in cats])
    ax.set_ylabel("number of cells"); ax.set_xlabel("responsive to stimulus")
    ax.set_title(f"Reversal significant responders ({df.session.nunique()} sessions; "
                 f"pre solid, post faded)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# --------------------------------------------------------------------------- #
# Responsiveness heatmaps (effect size = responsiveness, sorted, hot cmap)
# --------------------------------------------------------------------------- #
# Heatmaps display the avg stimulus response (trial-averaged mean dF/F in the
# 0-2 s stimulus window, baseline-subtracted; column `resp` in the long CSVs).
HEAT_VALUE = "resp"


def _rev_matrix(sdf):
    """neuron x (phase|stimulus) avg-stimulus-response matrix for a reversal frame."""
    sdf = sdf.copy()
    sdf["cond"] = sdf.phase.astype(str) + "|" + sdf.stimulus.astype(str)
    mat = sdf.pivot_table(index=["session", "neuron_id"], columns="cond", values=HEAT_VALUE)
    cols = [f"{ph}|{st}" for ph in ["pre", "post"] for st in REV_COND_ORDER]
    return mat.reindex(columns=[c for c in cols if c in mat.columns])


def _exp_matrix(sdf):
    mat = sdf.pivot_table(index=["session", "neuron_id"], columns="stimulus", values=HEAT_VALUE)
    return mat.reindex(columns=[c for c in EXP_ORDER if c in mat.columns])


def _draw_matrix(ax, mat, labels, colorbar, fig, title, ylabel_fs=17, xfs=None,
                 sig_mask=None, nonresp_label_fs=None):
    """Draw the neuron x condition heatmap. If ``sig_mask`` is given (bool array,
    one per row of ``mat``, True = significant responder to >=1 condition shown),
    responders and non-responders are drawn as two separate blocks -- responders
    on top (grouped/sorted by preferred condition as before), non-responders
    below a solid divider line (sorted the same way among themselves, purely for
    a tidy visual gradient -- their group membership is not meaningful since
    they're not significant). Without ``sig_mask`` all rows are grouped/sorted
    together as before (unchanged behaviour)."""
    M = mat.values
    m = np.nan_to_num(M)
    pref = np.argmax(m, axis=1)
    peak = m[np.arange(len(m)), pref]
    if sig_mask is None:
        order = np.lexsort((-peak, pref))
        n_resp = len(order)
    else:
        sig_mask = np.asarray(sig_mask, dtype=bool)
        assert len(sig_mask) == len(m), "sig_mask must have one entry per heatmap row"
        resp_idx = np.where(sig_mask)[0]
        nonresp_idx = np.where(~sig_mask)[0]
        resp_order = resp_idx[np.lexsort((-peak[resp_idx], pref[resp_idx]))] if len(resp_idx) else resp_idx
        # Non-responders are NOT grouped by preferred stimulus -- that sorting
        # implies tuning structure that didn't reach significance, which would
        # defeat the point of separating them out. Just a plain peak-descending
        # gradient (their "peak" is noise, but it's a harmless, non-misleading
        # ordering).
        nonresp_order = (nonresp_idx[np.argsort(-peak[nonresp_idx])]
                         if len(nonresp_idx) else nonresp_idx)
        order = np.concatenate([resp_order, nonresp_order])
        n_resp = len(resp_order)
    vmax = np.nanpercentile(np.abs(M), 99) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M[order], aspect="auto", cmap=HOT, vmin=0, vmax=vmax, interpolation="nearest")
    resp_block = order[:n_resp] if sig_mask is not None else order
    for b in np.where(np.diff(pref[resp_block]) != 0)[0]:
        ax.axhline(b + 0.5, color="white", lw=0.8, ls=(0, (1, 1.3)), alpha=0.9)
    if sig_mask is not None and 0 < n_resp < len(order):
        # Solid divider between the responder block (above) and non-responders
        # (below) -- a text label was tried here too but collided with the
        # colorbar/its label at every panel size this is used at (pooled,
        # per-session, and the small multi-session grid), so the divider line
        # alone carries the separation; a "responders above the line, n=...,
        # non-responders below" convention is noted in each figure's caption/
        # title instead where there's room.
        ax.axhline(n_resp - 0.5, color="black", lw=1.4, ls="-", zorder=5)
    if xfs is None:
        xfs = 11 if mat.shape[1] > 3 else 15
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(labels[:mat.shape[1]], fontsize=xfs)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("neuron (grouped, sorted)", fontsize=ylabel_fs)
    if title:
        ax.set_title(title, fontsize=16)
    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("avg stimulus response ($\\Delta$F/F)", fontsize=15)
        cb.ax.tick_params(labelsize=13)
    return im


def _rev_labels():
    return [f"{st}\n{ph}" for ph in ["pre", "post"] for st in ["100→0", "50", "0→100"]]


def draw_session_heatmap(scope, session, ax=None, colorbar=True, exclude_nonresp=False):
    """Single-session neuron x condition responsiveness heatmap.

    ``exclude_nonresp=True`` drops non-responding cells entirely (no
    divider/lower block) instead of the default of showing them below a
    divider line."""
    fig, ax, _ = new_panel(ax, figsize=(3.4, 5))
    long = _exp_long() if scope == "expert" else _rev_long()
    df = pd.read_csv(SUB / long)
    df["stimulus"] = df["stimulus"].astype(str)
    sdf = df[df.session == session]
    if scope == "reversal":
        mat = _rev_matrix(sdf); labels = _rev_labels()
    else:
        mat = _exp_matrix(sdf); labels = ["100%", "50%", "0%"]
    sig_mask = _any_sig(sdf, mat)
    if exclude_nonresp:
        mat = mat[sig_mask]
        sig_mask = None
    _draw_matrix(ax, mat, labels, colorbar, fig,
                 session.split("_")[-1] + " " + session[:10],
                 sig_mask=sig_mask)
    return fig


def draw_pooled_heatmap(scope, ax=None, colorbar=True, exclude_nonresp=False):
    """All cells from all sessions pooled into one neuron x condition heatmap.

    ``exclude_nonresp=True`` drops non-responding cells entirely (no
    divider/lower block) instead of the default of showing them below a
    divider line."""
    fig, ax, _ = new_panel(ax, figsize=(3.6, 5.2))
    long = _exp_long() if scope == "expert" else _rev_long()
    df = pd.read_csv(SUB / long)
    df["stimulus"] = df["stimulus"].astype(str)
    if scope == "reversal":
        mat = _rev_matrix(df); labels = _rev_labels()
    else:
        mat = _exp_matrix(df); labels = ["100%", "50%", "0%"]
    n = mat.shape[0]
    sig_mask = _any_sig(df, mat)
    n_resp = int(sig_mask.sum())
    if exclude_nonresp:
        mat = mat[sig_mask]
        title = f"responders only (n={n_resp})"
        sig_mask_arg = None
    else:
        title = f"all cells pooled (n={n}; {n_resp} responders above divider)"
        sig_mask_arg = sig_mask
    _draw_matrix(ax, mat, labels, colorbar, fig, title, sig_mask=sig_mask_arg)
    ax.set_yticks([])
    return fig


def draw_expert_heatmap_pooled(ax=None, exclude_nonresp=False):
    return draw_pooled_heatmap("expert", ax, exclude_nonresp=exclude_nonresp)


def draw_reversal_heatmap_pooled(ax=None, exclude_nonresp=False):
    return draw_pooled_heatmap("reversal", ax, exclude_nonresp=exclude_nonresp)


def _session_grid(scope, title, exclude_nonresp=False):
    long = _exp_long() if scope == "expert" else _rev_long()
    df = pd.read_csv(SUB / long)
    df["stimulus"] = df["stimulus"].astype(str)
    sessions = sorted(df.session.unique())
    ncols = 4
    nrows = -(-len(sessions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.3 * nrows), squeeze=False)
    labels = _rev_labels() if scope == "reversal" else ["100%", "50%", "0%"]
    im = None
    for i, sess in enumerate(sessions):
        ax = axes[i // ncols][i % ncols]
        sdf = df[df.session == sess]
        mat = _rev_matrix(sdf) if scope == "reversal" else _exp_matrix(sdf)
        sig_mask = _any_sig(sdf, mat)
        if exclude_nonresp:
            mat = mat[sig_mask]
            sig_mask = None
        im = _draw_matrix(ax, mat, labels, False, fig,
                          sess.split("_")[-1] + " " + sess[:10], ylabel_fs=7,
                          xfs=5, sig_mask=sig_mask, nonresp_label_fs=4)
        ax.title.set_fontsize(7)
        ax.set_ylabel("neuron (grouped, sorted)", fontsize=7)
    for j in range(len(sessions), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="responsiveness (z)")
    fig.suptitle(title + (" -- responders only" if exclude_nonresp else ""), fontsize=11)
    return fig


def draw_expert_heatmaps(ax=None, exclude_nonresp=False):
    return _session_grid("expert", "Expert responsiveness heatmaps (sorted; hot)", exclude_nonresp=exclude_nonresp)


def draw_reversal_heatmaps(ax=None, exclude_nonresp=False):
    return _session_grid("reversal", "Reversal responsiveness heatmaps (sorted; hot)", exclude_nonresp=exclude_nonresp)


# --------------------------------------------------------------------------- #
# Sankey (pre -> post responder-group transitions)
# --------------------------------------------------------------------------- #
def _reversal_groups(g_long, sessions=None):
    """Collapse the per-stimulus long table to one node per cell per phase:
    sig_{phase} = responder to ANY stimulus in that phase; pref_{phase} = the
    stimulus with the largest avg stimulus response among those it significantly
    responds to (falls back to overall max response when none is significant, but
    then sig is False so it is treated as non-responsive)."""
    df = g_long.copy()
    df["stimulus"] = df["stimulus"].astype(str); df["phase"] = df["phase"].astype(str)
    if sessions is not None:
        df = df[df.session.isin(sessions)]
    rows = []
    for (sess, nid), cell in df.groupby(["session", "neuron_id"]):
        rec = {"session": sess, "neuron_id": nid}
        for phase in ["pre", "post"]:
            p = cell[cell.phase == phase]
            sig_rows = p[p.sig]
            src = sig_rows if len(sig_rows) else p
            if len(src):
                pref = src.loc[src.resp.idxmax(), "stimulus"]
            else:
                pref = REV_ORDER[0]
            rec[f"sig_{phase}"] = bool(len(sig_rows) > 0)
            rec[f"pref_{phase}"] = pref
        rows.append(rec)
    return pd.DataFrame(rows)


def _transition_matrix(g, sessions=None, exclude_nonresp=False):
    """pre->post transition matrix over {responsive stimuli} + non-resp.

    When ``exclude_nonresp`` is True the non-responsive node is kept, but cells
    that *stay* non-responsive in both phases are dropped — so the node only
    carries cells that gain (non-resp pre -> responsive post) or lose
    (responsive pre -> non-resp post) responsiveness."""
    if sessions is not None:
        g = g[g.session.isin(sessions)]
    cats = REV_ORDER + [NONRESP]
    idx = {c: i for i, c in enumerate(cats)}
    left = np.where(g.sig_pre, g.pref_pre.astype(str), NONRESP)
    right = np.where(g.sig_post, g.pref_post.astype(str), NONRESP)
    M = np.zeros((len(cats), len(cats)))
    for l, r in zip(left, right):
        if exclude_nonresp and l == NONRESP and r == NONRESP:
            continue
        if l in idx and r in idx:
            M[idx[l], idx[r]] += 1
    return M, cats


def _bezier_band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.45):
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatches
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha))


def _draw_sankey(ax, M, cats, title, col_map=None, pretty_map=None):
    col_map = GROUP_COL if col_map is None else col_map
    pretty_map = PRETTY if pretty_map is None else pretty_map
    total = M.sum()
    gap = 0.02 * total if total else 1
    node_w = 0.13

    def extents(sizes):
        ys = {}
        y = 0.0
        for i, s in enumerate(sizes):
            ys[i] = (y, y + s)
            y += s + gap
        return ys, y
    left_sizes = M.sum(axis=1); right_sizes = M.sum(axis=0)
    Ly, Ltop = extents(left_sizes); Ry, Rtop = extents(right_sizes)
    top = max(Ltop, Rtop)
    for i, c in enumerate(cats):
        y0, y1 = Ly[i]
        ax.add_patch(plt.Rectangle((0, top - y1), node_w, y1 - y0, color=col_map[c]))
        ax.text(-0.02, top - (y0 + y1) / 2, pretty_map[c], ha="right", va="center", fontsize=12)
        y0, y1 = Ry[i]
        ax.add_patch(plt.Rectangle((1 - node_w, top - y1), node_w, y1 - y0, color=col_map[c]))
        ax.text(1.02, top - (y0 + y1) / 2, pretty_map[c], ha="left", va="center", fontsize=12)
    lcur = {i: Ly[i][0] for i in range(len(cats))}
    rcur = {j: Ry[j][0] for j in range(len(cats))}
    for i in range(len(cats)):
        for j in range(len(cats)):
            f = M[i, j]
            if f <= 0:
                continue
            y0a = top - lcur[i]; y0b = top - (lcur[i] + f); lcur[i] += f
            y1a = top - rcur[j]; y1b = top - (rcur[j] + f); rcur[j] += f
            _bezier_band(ax, node_w, y0a, y0b, 1 - node_w, y1a, y1b, col_map[cats[i]])
    ax.set_xlim(-0.28, 1.28); ax.set_ylim(-0.16 * top, top * 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    ax.text(node_w / 2, -0.09 * top, "pre", fontsize=18, ha="center", va="center")
    ax.text(1 - node_w / 2, -0.09 * top, "post", fontsize=18, ha="center", va="center")


def draw_sankey_combined(ax=None, exclude_nonresp=False):
    fig, ax, _ = new_panel(ax, figsize=(6, 6))
    g = _reversal_groups(pd.read_csv(SUB / _rev_long()))
    M, cats = _transition_matrix(g, exclude_nonresp=exclude_nonresp)
    _draw_sankey(ax, M, cats, f"Reversal responder pre→post ({g.session.nunique()} sessions combined)")
    return fig


def draw_sankey_persession(ax=None):
    g = _reversal_groups(pd.read_csv(SUB / _rev_long()))
    sessions = sorted(g.session.unique())
    ncols = 4
    nrows = -(-len(sessions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for i, sess in enumerate(sessions):
        M, cats = _transition_matrix(g, [sess])
        _draw_sankey(axes[i // ncols][i % ncols], M, cats, sess.split("_")[-1] + " " + sess[:10])
    for j in range(len(sessions), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Reversal responder pre→post Sankey, per session", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fine-grained combination groups (pure + mixed selectivity)
# --------------------------------------------------------------------------- #
# A cell is classified by the *set* of stimuli it significantly responds to:
# pure single-stimulus groups, mixed pairs (e.g. 0% & 50%), the all-three group,
# and non-responsive. This is a finer split of the per-stimulus size bars.
def _blend(c1, c2):
    import matplotlib.colors as mcol
    a, b = np.array(mcol.to_rgb(c1)), np.array(mcol.to_rgb(c2))
    return tuple((a + b) / 2.0)


def _fine_cats(order, include_nonresp=True):
    a, b, c = order
    cats = [a, b, c, f"{a}&{b}", f"{a}&{c}", f"{b}&{c}", f"{a}&{b}&{c}"]
    if include_nonresp:
        cats.append(NONRESP)
    return cats


def _fine_color(cat, order):
    if cat == NONRESP:
        return "0.7"
    parts = cat.split("&")
    if len(parts) == 1:
        return GROUP_COL[parts[0]]
    if len(parts) == 3:
        return "0.25"
    return _blend(GROUP_COL[parts[0]], GROUP_COL[parts[1]])


def _fine_pretty(cat):
    if cat == NONRESP:
        return "non-resp"
    parts = cat.split("&")
    if len(parts) == 3:
        return "all"
    return " &\n".join(PRETTY[p] for p in parts)


def _cell_category(sig_stims, order):
    """Canonical combination label for the set of stimuli a cell responds to."""
    S = [s for s in order if s in sig_stims]
    if not S:
        return NONRESP
    return "&".join(S)


def _fine_counts(df, order, phase=None):
    """Map each cell -> combination category and count per category."""
    if phase is not None:
        df = df[df.phase.astype(str) == phase]
    sig_sets = (df[df.sig].groupby(["session", "neuron_id"]).stimulus
                .apply(lambda s: set(s.astype(str))))
    all_cells = df[["session", "neuron_id"]].drop_duplicates()
    cats = []
    sig_lookup = sig_sets.to_dict()
    for _, (sess, nid) in all_cells.iterrows():
        cats.append(_cell_category(sig_lookup.get((sess, nid), set()), order))
    return pd.Series(cats).value_counts()


def draw_expert_sizes_fine(ax=None, include_nonresp=True):
    fig, ax, _ = new_panel(ax, figsize=(7, 4.6))
    df = pd.read_csv(SUB / _exp_long())
    df["stimulus"] = df["stimulus"].astype(str)
    cats = _fine_cats(EXP_ORDER, include_nonresp)
    vc = _fine_counts(df, EXP_ORDER)
    counts = [int(vc.get(c, 0)) for c in cats]
    ax.bar(range(len(cats)), counts, color=[_fine_color(c, EXP_ORDER) for c in cats])
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([_fine_pretty(c) for c in cats], fontsize=11)
    ax.set_ylabel("number of cells"); ax.set_xlabel("responsive to stimulus set")
    ax.set_title(f"Expert selectivity groups ({df.session.nunique()} sessions, "
                 f"{_n_cells(df)} neurons)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_reversal_sizes_fine(ax=None, include_nonresp=True):
    fig, ax, _ = new_panel(ax, figsize=(8.5, 4.6))
    df = pd.read_csv(SUB / _rev_long())
    df["stimulus"] = df["stimulus"].astype(str); df["phase"] = df["phase"].astype(str)
    cats = _fine_cats(REV_ORDER, include_nonresp)
    w = 0.38
    for phase in ["pre", "post"]:
        vc = _fine_counts(df, REV_ORDER, phase=phase)
        counts = [int(vc.get(c, 0)) for c in cats]
        off = -w / 2 if phase == "pre" else w / 2
        ax.bar(np.arange(len(cats)) + off, counts, width=w,
               color=[_fine_color(c, REV_ORDER) for c in cats],
               alpha=1.0 if phase == "pre" else 0.55,
               label="pre" if phase == "pre" else "post")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([_fine_pretty(c) for c in cats], fontsize=10)
    ax.set_ylabel("number of cells"); ax.set_xlabel("responsive to stimulus set")
    ax.set_title(f"Reversal selectivity groups ({df.session.nunique()} sessions; "
                 f"pre solid, post faded)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _fine_transition(df, exclude_nonresp=False):
    """pre->post transition matrix over the fine combination categories.

    When ``exclude_nonresp`` is True the non-responsive node is kept, but cells
    that stay non-responsive in both phases are dropped — so it only carries
    cells that gain or lose responsiveness."""
    df = df.copy()
    df["stimulus"] = df["stimulus"].astype(str); df["phase"] = df["phase"].astype(str)
    cats = _fine_cats(REV_ORDER, include_nonresp=True)
    idx = {c: i for i, c in enumerate(cats)}
    per = {}
    for phase in ["pre", "post"]:
        sub = df[df.phase == phase]
        sig_sets = (sub[sub.sig].groupby(["session", "neuron_id"]).stimulus
                    .apply(lambda s: set(s.astype(str))).to_dict())
        per[phase] = sig_sets
    cells = df[["session", "neuron_id"]].drop_duplicates()
    M = np.zeros((len(cats), len(cats)))
    for _, (sess, nid) in cells.iterrows():
        l = _cell_category(per["pre"].get((sess, nid), set()), REV_ORDER)
        r = _cell_category(per["post"].get((sess, nid), set()), REV_ORDER)
        if exclude_nonresp and l == NONRESP and r == NONRESP:
            continue
        if l in idx and r in idx:
            M[idx[l], idx[r]] += 1
    col = {c: _fine_color(c, REV_ORDER) for c in cats}
    pretty = {c: _fine_pretty(c).replace("\n", " ") for c in cats}
    return M, cats, col, pretty


def draw_sankey_fine_combined(ax=None, exclude_nonresp=False):
    fig, ax, _ = new_panel(ax, figsize=(6.5, 7))
    df = pd.read_csv(SUB / _rev_long())
    M, cats, col, pretty = _fine_transition(df, exclude_nonresp=exclude_nonresp)
    _draw_sankey(ax, M, cats, f"Reversal selectivity-group pre→post "
                 f"({df.session.nunique()} sessions)", col_map=col, pretty_map=pretty)
    return fig


def build_all(show_tag=None, methods=("temporal", "time_averaged")):
    global METHOD
    _orig_method = METHOD
    try:
        for method in methods:
            METHOD = method
            if not _has(_rev_long()):
                print(f"  (no {method} responder data for reversal — run the matching "
                      f"extract/extract_responsiveness_*.py script)")
                continue
            if _has(_exp_long()):
                save_panel(draw_expert_sizes(), f"Subgroups/Expert_{method}", f"SUB.EXP.sizes.{method}", f"expert_responder_sizes_{method}", show_tag)
                save_panel(draw_expert_sizes(include_nonresp=False), f"Subgroups/Expert_{method}", f"SUB.EXP.sizesresp.{method}", f"expert_responder_sizes_responders_{method}", show_tag)
                save_panel(draw_expert_sizes_fine(), f"Subgroups/Expert_{method}", f"SUB.EXP.sizesfine.{method}", f"expert_selectivity_groups_fine_{method}", show_tag)
                save_panel(draw_expert_sizes_fine(include_nonresp=False), f"Subgroups/Expert_{method}", f"SUB.EXP.sizesfineresp.{method}", f"expert_selectivity_groups_fine_responders_{method}", show_tag)
                save_panel(draw_expert_heatmaps(), f"Subgroups/Expert_{method}", f"SUB.EXP.heatmap.{method}", f"expert_responsiveness_heatmaps_{method}", show_tag)
                save_panel(draw_expert_heatmap_pooled(), f"Subgroups/Expert_{method}", f"SUB.EXP.heatmappool.{method}", f"expert_responsiveness_heatmap_pooled_{method}", show_tag)
                save_panel(draw_expert_heatmap_pooled(exclude_nonresp=True), f"Subgroups/Expert_{method}", f"SUB.EXP.heatmappoolresp.{method}", f"expert_responsiveness_heatmap_pooled_responders_{method}", show_tag)
            save_panel(draw_reversal_sizes(), f"Subgroups/Reversal_{method}", f"SUB.REV.sizes.{method}", f"reversal_responder_sizes_{method}", show_tag)
            save_panel(draw_reversal_sizes(include_nonresp=False), f"Subgroups/Reversal_{method}", f"SUB.REV.sizesresp.{method}", f"reversal_responder_sizes_responders_{method}", show_tag)
            save_panel(draw_reversal_sizes_fine(), f"Subgroups/Reversal_{method}", f"SUB.REV.sizesfine.{method}", f"reversal_selectivity_groups_fine_{method}", show_tag)
            save_panel(draw_reversal_sizes_fine(include_nonresp=False), f"Subgroups/Reversal_{method}", f"SUB.REV.sizesfineresp.{method}", f"reversal_selectivity_groups_fine_responders_{method}", show_tag)
            save_panel(draw_reversal_heatmaps(), f"Subgroups/Reversal_{method}", f"SUB.REV.heatmap.{method}", f"reversal_responsiveness_heatmaps_{method}", show_tag)
            save_panel(draw_reversal_heatmap_pooled(), f"Subgroups/Reversal_{method}", f"SUB.REV.heatmappool.{method}", f"reversal_responsiveness_heatmap_pooled_{method}", show_tag)
            save_panel(draw_reversal_heatmap_pooled(exclude_nonresp=True), f"Subgroups/Reversal_{method}", f"SUB.REV.heatmappoolresp.{method}", f"reversal_responsiveness_heatmap_pooled_responders_{method}", show_tag)
            save_panel(draw_sankey_persession(), f"Subgroups/Reversal_{method}", f"SUB.REV.sankey.{method}", f"reversal_sankey_persession_{method}", show_tag)
            save_panel(draw_sankey_combined(), f"Subgroups/Reversal_{method}", f"SUB.REV.sankeycomb.{method}", f"reversal_sankey_combined_{method}", show_tag)
            save_panel(draw_sankey_fine_combined(), f"Subgroups/Reversal_{method}", f"SUB.REV.sankeyfine.{method}", f"reversal_sankey_fine_combined_{method}", show_tag)
            save_panel(draw_sankey_fine_combined(exclude_nonresp=True), f"Subgroups/Reversal_{method}", f"SUB.REV.sankeyfineresp.{method}", f"reversal_sankey_fine_responders_{method}", show_tag)
    finally:
        METHOD = _orig_method


if __name__ == "__main__":
    build_all()
