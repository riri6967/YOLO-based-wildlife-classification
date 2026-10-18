"""Run the model x threshold x season x monitoring-regime comparison.

    python scripts/run_experiment.py --reps 20 --out results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildsim import load_config, plots  # noqa: E402
from wildsim.experiment import run  # noqa: E402

SUMMARY_COLS = [
    "recall_day", "recall_night", "recall_open_veg", "recall_dense_veg", "recall_rain",
    "mae_occ_naive", "mae_occ_mackenzie", "mae_occ_royle_link", "site_richness_bias",
    "missed_event_frac", "false_alarms", "median_latency_h", "timely_response_frac",
    "compute_kwh", "co2_inference_kg", "co2_training_kg", "co2_field_kg", "co2_total_kg",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    scen, spec = run(cfg, thresholds=tuple(a.thresholds), n_reps=a.reps, seed=a.seed)
    scen.to_csv(out / "scenario_results.csv", index=False)
    spec.to_csv(out / "species_results.csv", index=False)

    summary = scen.groupby(["season", "regime", "model", "threshold"])[SUMMARY_COLS].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")

    plots.occupancy_bias(spec, out)
    plots.environmental_recall(scen, out)
    plots.threshold_tradeoff(scen, out)
    plots.carbon_vs_accuracy(scen, out)

    view = scen[scen.threshold == 0.5].groupby(["season", "regime", "model"])[
        ["recall_night", "recall_dense_veg", "mae_occ_mackenzie", "mae_occ_royle_link",
         "missed_event_frac", "timely_response_frac", "co2_total_kg"]].mean().round(3)
    print(view.to_string())
    print(f"\nWrote results and figures to {out.resolve()}")


if __name__ == "__main__":
    main()
