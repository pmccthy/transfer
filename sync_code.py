#!/usr/bin/env python3
"""Sync CODE (as opposed to data/figs) from the three source repos into this
repo's top-level extract/, analysis/, and fig_gen/ directories, flattened
(no per-mirror subfolders) with filenames disambiguated only where two
mirrors actually collide.

Companion to sync_rnns.sh / sync_experimental.sh / sync_combined.sh, which
now sync DATA ONLY into data/{rnns,experimental,combined}/ (their code
subfolders are excluded there -- see the EXCLUDES lists in each of those
scripts). Run via sync_all.sh, or standalone: `python3 sync_code.py`.

IMPORTANT -- these are REFERENCE COPIES, not a runnable working copy. Each
source file's internal sys.path.insert()/relative-path logic is written
assuming it still sits in its ORIGINAL location inside the source repo
(e.g. combined/panels/chi2_bars.py expects a sibling ../analysis/ and
../cross_model_vs_experiment/group_counts/ two levels up from panels/) --
none of that is rewritten by this script. To actually RUN any of this code,
use it from its home repo:
    context-value-RNNs/results/transfer/{code,combined}/...
    neuronal-representations/results/transfer/{extract,figures}/...
This copy exists so the code is browsable and version-controlled as
first-class content of this repo, organized by ROLE (what it does) rather
than buried inside the gitignored data/ sync mirrors.

Filenames get a _combined / _experimental suffix ONLY where two mirrors
used the identical filename (see COLLISIONS below) -- everything else keeps
its original name.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys
from pathlib import Path

DRY = any(a in ("--dry-run", "-n") for a in sys.argv[1:])

HERE = Path(__file__).resolve().parent
CXVAL = Path(os.environ.get("CXVAL_REPO", str(Path.home() / "Documents" / "context-value-RNNs")))
NEURONAL = Path(os.environ.get("NEURONAL_REPO", str(Path.home() / "Documents" / "neuronal-representations")))

RNNS_CODE = CXVAL / "results" / "transfer" / "code"
RNNS_REVERSAL_CODE = CXVAL / "results" / "transfer" / "reversal" / "code"
COMBINED = CXVAL / "results" / "transfer" / "combined"
EXPERIMENTAL = NEURONAL / "results" / "transfer"

SKIP_NAMES = {"__pycache__", ".DS_Store"}
SKIP_SUFFIXES = {".pyc"}


def _files(d: Path):
    if not d.is_dir():
        return
    for p in sorted(d.iterdir()):
        if p.name in SKIP_NAMES or p.suffix in SKIP_SUFFIXES or p.is_dir():
            continue
        if p.name.startswith(".") or p.name.startswith("~"):
            continue  # editor/FUSE temp files (.fuse_hidden*, .DS_Store, ~lock)
        yield p


# (source_path, category, dest_filename) -- dest_filename renamed only on
# an actual cross-mirror collision (see module docstring).
MAPPING = []

# -- rnns/code/ -------------------------------------------------------------
MAPPING += [
    (RNNS_CODE / "build_figure_data_from_timeresolved.py", "extract", None),
    (RNNS_CODE / "compare_responder_definitions.py", "analysis", None),
    (RNNS_CODE / "reversal_analysis.py", "analysis", None),
    (RNNS_CODE / "figure_config.py", "fig_gen", None),
    (RNNS_CODE / "figure_style.mplstyle", "fig_gen", None),
    (RNNS_CODE / "figures.py", "fig_gen", None),
    (RNNS_CODE / "explore_figure_data.ipynb", "notebooks", None),
    (RNNS_CODE / "explore_time_resolved.ipynb", "notebooks", None),
    (RNNS_CODE / "make_figures.ipynb", "notebooks", None),
]

# -- rnns/reversal/code/ -- the reversal-study's OWN analysis/fig-gen code,
# separate from rnns/code/ above (the pre-reversal/expert-side pipeline).
# figure_config.py and figure_style.mplstyle are byte-identical to rnns/code's
# copies (checked) -- not re-copied, one shared copy in fig_gen/ covers both.
# figures.py and reversal_analysis.py DIFFER from rnns/code's same-named
# files (checked) -- these are the reversal-study's own variants, suffixed
# "_reversal"/"_study" to keep both distinguishable. run_*_full.sh /
# run_reward_scale_*.sh sweep launchers are left in data/rnns/reversal/code/
# untouched -- they're reproducibility/training material (see rnns/README.md),
# not extract/analysis/fig_gen code.
MAPPING += [
    (RNNS_REVERSAL_CODE / "figures.py", "fig_gen", "figures_reversal.py"),
    (RNNS_REVERSAL_CODE / "reversal_analysis.py", "analysis", "reversal_analysis_study.py"),
    (RNNS_REVERSAL_CODE / "make_reward_scale_paired_figure.py", "fig_gen", None),
    (RNNS_REVERSAL_CODE / "population_similarity.py", "analysis", None),
    (RNNS_REVERSAL_CODE / "recovery_time.py", "analysis", None),
    (RNNS_REVERSAL_CODE / "reversal_onset_probe.py", "analysis", None),
    (RNNS_REVERSAL_CODE / "rpe_asymmetry_bar.py", "fig_gen", None),
    (RNNS_REVERSAL_CODE / "rsa.py", "analysis", None),
    (RNNS_REVERSAL_CODE / "seed_groups.py", "analysis", None),
    (RNNS_REVERSAL_CODE / "terminal_rpe.py", "analysis", None),
]

# -- combined/analysis/ -------------------------------------------------------
for p in _files(COMBINED / "analysis"):
    MAPPING.append((p, "analysis", None))

# -- combined/cross_model_vs_experiment/  (code only -- group_counts/ +
#    README.md stay behind as data, synced by sync_combined.sh) -------------
for p in _files(COMBINED / "cross_model_vs_experiment"):
    if p.suffix == ".py":
        MAPPING.append((p, "analysis" if p.name == "chi2_metrics.py" else "extract", None))

# -- combined/panels/, combined/style/, combined/compose.py,
#    combined/make_panels.py -> fig_gen/ ------------------------------------
for p in _files(COMBINED / "panels"):
    rename = {"_tags.py": "_tags_combined.py", "decoding.py": "decoding_combined.py"}.get(p.name)
    MAPPING.append((p, "fig_gen/panels", rename))
for p in _files(COMBINED / "style"):
    MAPPING.append((p, "fig_gen", None))
MAPPING.append((COMBINED / "compose.py", "fig_gen", "compose_combined.py"))
MAPPING.append((COMBINED / "make_panels.py", "fig_gen", "make_panels_combined.py"))

# -- experimental (neuronal-representations results/transfer/) --------------
for p in _files(EXPERIMENTAL / "extract"):
    MAPPING.append((p, "extract", None))
for p in _files(EXPERIMENTAL / "figures"):
    rename = {"compose.py": "compose_experimental.py", "make_panels.py": "make_panels_experimental.py"}.get(p.name)
    MAPPING.append((p, "fig_gen", rename))
for p in _files(EXPERIMENTAL / "figures" / "panels"):
    rename = {"_tags.py": "_tags_experimental.py", "decoding.py": "decoding_experimental.py"}.get(p.name)
    MAPPING.append((p, "fig_gen/panels", rename))


def main():
    missing = [str(src) for src, _, _ in MAPPING if not src.exists()]
    if missing:
        print(f"!! {len(missing)} source file(s) not found (source repo layout may have moved):")
        for m in missing[:20]:
            print(f"   {m}")

    by_dest_dir = {}
    n_copied = n_same = n_would = 0
    for src, category, rename in MAPPING:
        if not src.exists():
            continue
        dest_dir = HERE / category
        dest = dest_dir / (rename or src.name)
        by_dest_dir.setdefault(category, []).append(dest.name)
        if dest.exists() and filecmp.cmp(src, dest, shallow=False):
            n_same += 1
            continue
        if DRY:
            print(f"  [dry-run] {src.relative_to(src.parents[3]) if len(src.parents) > 3 else src} -> {category}/{dest.name}")
            n_would += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        n_copied += 1

    # collision sanity check: any duplicate dest filename within one category
    # means a mirror was added/renamed and MAPPING needs another entry above.
    for category, names in by_dest_dir.items():
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            print(f"!! collision in {category}/: {dupes} -- add a rename in MAPPING")

    if DRY:
        print(f"\n{n_would} file(s) would be copied, {n_same} already up to date.")
    else:
        print(f"\n{n_copied} file(s) copied/updated, {n_same} already up to date.")


if __name__ == "__main__":
    main()
