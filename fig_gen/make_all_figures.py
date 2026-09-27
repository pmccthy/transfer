"""Regenerate every transfer figure from the tidy CSVs.

Run from anywhere:  python figures/make_all_figures.py
Figures are written to figures/figs/output/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import plot_tdr
import plot_pca
import plot_decoding
import plot_population_means
import plot_heatmaps
import plot_lick


def main():
    plot_tdr.main()
    plot_pca.main()
    plot_decoding.main()
    plot_population_means.main()
    plot_heatmaps.main()
    plot_lick.main()
    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
