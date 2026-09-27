"""Neuron subsampling cap for the decoding extractors.

Matches the canonical SAT decoding pipeline, which decodes from a fixed number
of neurons (``num_neurons``) subsampled from each session's SNR-filtered set,
averaged over several random neuron permutations. This equalises neuron count
across sessions and keeps decoding accuracies comparable to the blessed run
(using *all* SNR neurons gives systematically higher accuracy — see the decoding
discrepancy investigation).

Config (overridable by env var so it is easy to switch back to all-neuron):
  TRANSFER_DECODE_NUM_NEURONS   cap per session (default 100; <=0 or "all" = no cap)
  TRANSFER_DECODE_NEURON_PERMS  number of random neuron subsets to average (default 10)
  TRANSFER_DECODE_NEURON_SEED   RNG seed for the subsets (default 42)

A session with fewer neurons than the cap uses all of them (one "permutation").
"""

from __future__ import annotations

import os

import numpy as np

_raw = os.environ.get("TRANSFER_DECODE_NUM_NEURONS", "100").strip().lower()
NUM_NEURONS = None if _raw in ("all", "none", "0", "-1", "") else int(_raw)
N_NEURON_PERM = int(os.environ.get("TRANSFER_DECODE_NEURON_PERMS", "10"))
NEURON_RANDOM_STATE = int(os.environ.get("TRANSFER_DECODE_NEURON_SEED", "42"))


def cap_enabled() -> bool:
    return NUM_NEURONS is not None and NUM_NEURONS > 0


def neuron_subsets(n_neurons: int):
    """List of neuron-index arrays to average a decoder over.

    - cap disabled, or session has <= cap neurons -> a single full-index array
      (i.e. behave exactly like the all-neuron path).
    - otherwise -> ``N_NEURON_PERM`` random size-``NUM_NEURONS`` subsets.
    """
    if not cap_enabled() or n_neurons <= NUM_NEURONS:
        return [np.arange(n_neurons)]
    rng = np.random.default_rng(NEURON_RANDOM_STATE)
    return [rng.choice(n_neurons, size=NUM_NEURONS, replace=False)
            for _ in range(N_NEURON_PERM)]


def average_time_resolved(perm_results):
    """Average a list of ``process_session`` outputs over neuron permutations.

    Each element is ``{decoder: {'time_ax','acc'}}`` (within) or
    ``{decoder: {'time_ax','train_acc','test_acc'}}`` (cross). Returns one dict
    of the same shape with the accuracy arrays averaged across permutations.
    """
    perm_results = [r for r in perm_results if r is not None]
    if not perm_results:
        return None
    if len(perm_results) == 1:
        return perm_results[0]
    out = {}
    for dec in perm_results[0]:
        e0 = perm_results[0][dec]
        merged = {"time_ax": e0["time_ax"]}
        for key in ("acc", "train_acc", "test_acc"):
            if key in e0:
                merged[key] = np.mean([np.asarray(pr[dec][key]) for pr in perm_results], axis=0)
        out[dec] = merged
    return out


def meta() -> dict:
    return {
        "num_neurons_cap": (NUM_NEURONS if cap_enabled() else None),
        "n_neuron_permutations": (N_NEURON_PERM if cap_enabled() else 1),
        "neuron_random_state": NEURON_RANDOM_STATE,
        "neuron_subsampling": (
            f"subsample {NUM_NEURONS} neurons/session, averaged over {N_NEURON_PERM} "
            "random permutations (sessions with fewer use all); matches the SAT "
            "canonical decoding pipeline" if cap_enabled()
            else "no cap — all SNR-filtered neurons used"),
    }
