"""
pareto_postprocess.py

Takes results/pareto_frontier.csv and produces:
- results/pareto_frontier_feasible.csv  (only feasible rows)
- results/pareto_knee_candidates.csv    (top knee solutions using simple normalized distance)

Run:
python optimization/pareto_postprocess.py
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

IN_CSV = os.path.join(RESULTS_DIR, "pareto_frontier.csv")
OUT_FEAS = os.path.join(RESULTS_DIR, "pareto_frontier_feasible.csv")
OUT_KNEE = os.path.join(RESULTS_DIR, "pareto_knee_candidates.csv")


def normalize(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax == smin:
        return series * 0.0
    return (series - smin) / (smax - smin)


def main() -> None:
    df = pd.read_csv(IN_CSV)

    # Keep feasible only
    feas = df[df["feasible"] == 1].copy()
    feas.to_csv(OUT_FEAS, index=False)

    if len(feas) == 0:
        print("[WARN] No feasible points found.")
        return

    # Define objectives:
    # minimize cost, minimize emissions, maximize recycling
    # Convert recycling to minimization by using (1 - recycling)
    feas["cost_norm"] = normalize(feas["total_cost_usd"])
    feas["emissions_norm"] = normalize(feas["total_emissions_kgco2"])
    feas["recycling_norm"] = normalize(1.0 - feas["avg_recycling_eff"])

    # Knee heuristic: minimize distance to ideal point (0,0,0)
    feas["knee_score"] = np.sqrt(
        feas["cost_norm"] ** 2 +
        feas["emissions_norm"] ** 2 +
        feas["recycling_norm"] ** 2
    )

    knee = feas.sort_values("knee_score", ascending=True).head(10).copy()

    # Drop helper cols before saving
    knee_out = knee.drop(columns=["cost_norm", "emissions_norm", "recycling_norm"])
    knee_out.to_csv(OUT_KNEE, index=False)

    print(f"[OK] Feasible frontier saved: {OUT_FEAS}  (n={len(feas)})")
    print(f"[OK] Knee candidates saved:  {OUT_KNEE}")
    print("\nTop 5 knee candidates:")
    print(knee_out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
