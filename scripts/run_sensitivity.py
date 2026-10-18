"""Sobol global sensitivity analysis.

    python scripts/run_sensitivity.py --n-base 256 --out results/
(--n-base must be a power of 2; total runs = n_base * (n_params + 2))
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildsim import load_config, plots  # noqa: E402
from wildsim.sensitivity import run_sobol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-base", type=int, default=128)
    ap.add_argument("--model", default="YOLO11", help="profile whose remaining parameters are held fixed")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    si, runs = run_sobol(load_config(a.config), n_base=a.n_base, base_model=a.model, seed=a.seed)
    si.to_csv(out / "sobol_indices.csv", index=False)
    runs.to_csv(out / "sobol_runs.csv", index=False)
    plots.sobol_indices(si, out)
    print(si.pivot(index="parameter", columns="output", values="ST").round(2).to_string())


if __name__ == "__main__":
    main()
