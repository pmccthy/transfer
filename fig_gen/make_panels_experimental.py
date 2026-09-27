"""Build every individual figure panel into the hierarchical figures/ folders.

Each panel is a self-contained draw_*(ax=...) function (see panels/*.py) so it
can also be reused as a subpanel of a larger composite figure. Every panel is
saved as figures/<hierarchy>/<TAG>__<name>.png with its tag stamped on it.

Toggle the on-figure tag globally:
    python make_panels.py            # tags shown
    SHOW_TAG=0 python make_panels.py # tags hidden
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "panels"))

import _tags
_tags.SHOW_TAG = os.environ.get("SHOW_TAG", "1") not in ("0", "false", "False")

import behaviour
import population
import decoding
try:
    import decoding_pooled
except Exception:
    decoding_pooled = None
try:
    import subgroups
except Exception:
    subgroups = None
try:
    import dimensionality
except Exception:
    dimensionality = None
try:
    import coding_angle_experimental
except Exception:
    coding_angle_experimental = None


def main():
    print("Behaviour:");      behaviour.build_all()
    print("Population:");     population.build_all()
    print("Decoding (TR):");  decoding.build_all()
    if decoding_pooled:
        print("Decoding (TP):"); decoding_pooled.build_all()
    if subgroups:
        print("Subgroups:");  subgroups.build_all()
    if dimensionality:
        print("Dimensionality:"); dimensionality.build_all()
    if coding_angle_experimental:
        print("Coding-vector angles:"); coding_angle_experimental.build_all()
    print("\nPanels complete →", _tags.OUT_ROOT)


if __name__ == "__main__":
    main()
