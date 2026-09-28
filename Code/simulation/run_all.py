"""
Main entry point: run all simulation scenarios and generate figures.

Usage:
    python run_all.py                  # Full run (1000 reps)
    python run_all.py --pilot          # Pilot run (100 reps)
    python run_all.py --scenario s1    # Run a single scenario
"""

import sys
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenarios.s1_identification import run_s1
from scenarios.s2_surrogacy_violation import run_s2
from scenarios.s3_comparability_violation import run_s3
from scenarios.s4_sensitivity_bound import run_s4
from scenarios.s5_backtesting import run_s5
from scenarios.s6_multidraw_surrogacy import run_s6
from scenarios.s7_kdraw_estimation import run_s7
from scenarios.s9_consistency_rate import run_s9
from figures import (figure1_identification, figure2_failure_modes,
                     figure3_multidraw_surrogacy, figure4_estimation_quality,
                     figure5_sensitivity_bound, figure6_consistency_rate,
                     FIGDIR)


OUTDIR = Path(__file__).resolve().parent / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)


def save_results(results, name):
    path = OUTDIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {name} results to {path}")


def load_results(name):
    path = OUTDIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def run_scenarios(n_reps, seed=42, scenario=None):
    all_results = {}

    runners = {
        "s1": ("S1: Identification", lambda: run_s1(n_reps=n_reps, seed=seed)),
        "s2": ("S2: Surrogacy violation", lambda: run_s2(n_reps=n_reps, seed=seed)),
        "s3": ("S3: Comparability violation", lambda: run_s3(n_reps=n_reps, seed=seed)),
        "s4": ("S4: Sensitivity bound", lambda: run_s4(n_reps=min(n_reps, 200), seed=seed)),
        "s5": ("S5: Back-testing", lambda: run_s5(n_reps=n_reps, seed=seed)),
        "s6": ("S6: Multi-draw surrogacy", lambda: run_s6(n_reps=n_reps, seed=seed)),
        "s7": ("S7: K-draw estimation", lambda: run_s7(n_reps=n_reps, seed=seed)),
        "s9": ("S9: Consistency rate (n-sweep)", lambda: run_s9(n_reps=min(n_reps, 500), seed=seed)),
    }

    targets = [scenario] if scenario else list(runners.keys())

    for key in targets:
        label, runner = runners[key]
        print(f"\n{'='*60}")
        print(f"Running {label} ({n_reps} reps)...")
        print(f"{'='*60}")
        result = runner()
        all_results[key] = result
        save_results(result, key)

    return all_results


def generate_figures(results=None):
    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    def get(key):
        if results and key in results:
            return results[key]
        return load_results(key)

    figure1_identification(get("s1"))
    figure2_failure_modes(get("s2"), get("s3"))
    figure3_multidraw_surrogacy(get("s6"))
    figure4_estimation_quality(get("s7"))
    figure5_sensitivity_bound(get("s4"))
    try:
        figure6_consistency_rate(get("s9"))
    except FileNotFoundError:
        print("(skipping fig6_consistency_rate: s9.pkl not found; run --scenario s9)")

    print(f"\nAll figures saved to {FIGDIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot run with 100 reps")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a single scenario (s1-s7)")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only generate figures from saved results")
    args = parser.parse_args()

    if args.figures_only:
        generate_figures()
    else:
        n_reps = 100 if args.pilot else 1000
        results = run_scenarios(n_reps, scenario=args.scenario)
        if not args.scenario:
            generate_figures(results)
