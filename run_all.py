"""Rebuild everything: python run_all.py [--upto N]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")  # MacBook Air: keep sklearn to 2 threads
import argparse
import time

from src import data, checks

PHASES = {
    1: ("Data audit and cleaning", lambda: data.run()),
    2: ("Win probability model + endgame study", lambda: (__import__("src.wp_model", fromlist=["run"]).run(),
                                                          __import__("src.wp_endgame", fromlist=["run"]).run())),
    3: ("Pressure: leverage index", lambda: __import__("src.leverage", fromlist=["run"]).run()),
    4: ("Per-ball performance value", lambda: __import__("src.value", fromlist=["run"]).run()),
    5: ("Pressure metrics + role tags", lambda: (__import__("src.metrics", fromlist=["run"]).run(),
                                                 __import__("src.roles", fromlist=["run"]).run())),
    6: ("Repeatability, target, weights + 6B signal recovery",
        lambda: (__import__("src.repeatability", fromlist=["run"]).run(),
                 __import__("src.signal_recovery", fromlist=["run"]).run())),
    7: ("Rating and ranking", lambda: __import__("src.rating", fromlist=["run"]).run()),
    8: ("Validation", lambda: __import__("src.validate", fromlist=["run"]).run()),
    9: ("RR layer, outputs and FINAL.md", lambda: (__import__("src.outputs", fromlist=["run"]).run(),
                                                   __import__("src.final_report", fromlist=["run"]).run())),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=max(PHASES))
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    todo = [a.only] if a.only else [p for p in sorted(PHASES) if p <= a.upto]
    for p in todo:
        name, fn = PHASES[p]
        t = time.time()
        print(f"=== Phase {p}: {name}")
        fn()
        checks.run(p)                       # stops the pipeline if a sign/sanity check fails
        print(f"=== Phase {p} done in {time.time() - t:.1f}s\n")
