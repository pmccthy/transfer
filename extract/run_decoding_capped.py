#!/usr/bin/env python3
"""Run the (100-neuron-capped) decoding extractors with live progress + ETA.

Runs the three decoding extractors in turn:
  1. extract_decoding.py             reversal time-resolved
  2. extract_decoding_timepooled.py  reversal + expert time-pooled
  3. extract_decoding_expert.py      expert time-resolved
then regenerates the figures (make_panels.py + compose.py).

Progress/ETA is derived from each extractor's per-session cache: every session
writes one cache pkl when it finishes, so we poll the cache directory and report
``done/total`` plus elapsed and estimated-remaining time.

RESUMABLE: the per-session caches persist, so you can Ctrl-C this at any point
and rerun it — already-finished sessions are skipped instantly and it carries on
from where it stopped. Nothing computed so far is lost.

Cache location: ``$TRANSFER_DECODE_CACHE`` if set, else
``~/.cache/transfer_decode_cap100`` (the dir used by the capped run). Point it at
whatever your current run used so this reuses that progress.

Usage:
    python results/transfer/extract/run_decoding_capped.py
    python results/transfer/extract/run_decoding_capped.py --skip-figures
    TRANSFER_DECODE_CACHE=/path/to/cache python .../run_decoding_capped.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # neuronal-representations/
FIGS = HERE.parent / "figures"

# Default cache dir = the one the capped run used, so progress is reused.
os.environ.setdefault("TRANSFER_DECODE_CACHE",
                      str(Path.home() / ".cache" / "transfer_decode_cap100"))
CACHE = Path(os.environ["TRANSFER_DECODE_CACHE"])

sys.path.insert(0, str(HERE))
import _neuron_cap as NCAP  # noqa: E402

TAG = (f"cap{NCAP.NUM_NEURONS}x{NCAP.N_NEURON_PERM}"
       if NCAP.cap_enabled() else "allneurons")


def _session_count(kind: str, default: int) -> int:
    try:
        if kind == "reversal":
            from _sources import TDR_DIR  # noqa
            return len(json.loads((TDR_DIR / "analysis_summary_v2.json")
                                  .read_text())["session_names"])
        if kind == "expert":
            d = json.loads((REPO / "metadata" / "SAT_good_expert_sessions.json").read_text())
            return len(d if isinstance(d, list) else d.get("sessions", d))
    except Exception:
        pass
    return default


N_REV = _session_count("reversal", 13)
N_EXP = _session_count("expert", 44)

# (label, extractor script, cache subdir, expected #sessions)
STAGES = [
    ("reversal time-resolved", "extract_decoding.py", CACHE / TAG, N_REV),
    ("time-pooled (rev+expert)", "extract_decoding_timepooled.py",
     CACHE / f"timepooled_{TAG}", N_REV + N_EXP),
    ("expert time-resolved", "extract_decoding_expert.py",
     CACHE / f"expert_{TAG}", N_EXP),
]

POLL_S = 10
LOG_DIR = CACHE / "logs"


def fmt(sec: float) -> str:
    sec = int(max(sec, 0))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def n_cached(d: Path) -> int:
    return len(glob.glob(str(d / "*.pkl")))


def run_stage(label, script, cdir, total):
    cdir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logpath = LOG_DIR / f"{script}.log"
    start_done = n_cached(cdir)
    print(f"\n=== {label}  ({script}) ===")
    print(f"    already done (resumed): {start_done}/{total}   log: {logpath}")
    t0 = time.time()
    with open(logpath, "ab") as log:
        log.write(f"\n\n===== {time.ctime()}  run_decoding_capped: {script} =====\n".encode())
        log.flush()
        p = subprocess.Popen([sys.executable, "-u", str(HERE / script)],
                             stdout=log, stderr=subprocess.STDOUT, env=os.environ)
        while p.poll() is None:
            time.sleep(POLL_S)
            done = n_cached(cdir)
            new = done - start_done
            el = time.time() - t0
            if new > 0 and done < total:
                eta = (el / new) * (total - done)
                print(f"    {done}/{total} sessions | elapsed {fmt(el)} | "
                      f"~{fmt(eta)} left  (~{el/new:.0f}s/session)", flush=True)
            else:
                print(f"    {done}/{total} sessions | elapsed {fmt(el)} | "
                      f"{'finishing…' if done >= total else 'estimating…'}", flush=True)
    rc = p.returncode
    print(f"    {label}: DONE {n_cached(cdir)}/{total} in {fmt(time.time() - t0)}"
          f"{'' if rc == 0 else f'  [exit {rc} — see log]'}", flush=True)
    return rc


def main():
    skip_figs = "--skip-figures" in sys.argv
    print(f"Cache: {CACHE}\nNeuron cap: "
          f"{'%d neurons x %d perms' % (NCAP.NUM_NEURONS, NCAP.N_NEURON_PERM) if NCAP.cap_enabled() else 'ALL neurons'}")
    print(f"Sessions: reversal={N_REV}, expert={N_EXP}")
    t0 = time.time()
    bad = 0
    for stage in STAGES:
        bad += 1 if run_stage(*stage) not in (0, None) else 0
    print(f"\nAll decoding extractors finished in {fmt(time.time() - t0)}"
          f"{'' if not bad else f'  ({bad} stage(s) had non-zero exit)'}")

    # provenance + figures
    subprocess.run([sys.executable, str(HERE / "make_provenance.py")], check=False)
    if not skip_figs:
        print("\nRegenerating panels + composites…")
        env = {**os.environ, "MPLBACKEND": "Agg"}
        subprocess.run([sys.executable, "make_panels.py"], cwd=str(FIGS), env=env, check=False)
        subprocess.run([sys.executable, "compose.py"], cwd=str(FIGS), env=env, check=False)
        print("Figures regenerated →", FIGS / "composites")
    print("\nDone.")


if __name__ == "__main__":
    main()
