"""
pareto_table_export.py

Creates manuscript-ready Pareto tables from results/pareto_frontier_feasible.csv:

Outputs:
- results/pareto_table_top10.csv   (10 diverse frontier points)
- results/pareto_table_knee.csv    (single knee point)

Run:
python optimization/pareto_table_export.py
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

FEAS_CSV = os.path.join(RESULTS_DIR, "pareto_frontier_feasible.csv")
KNEE_CSV = os.path.join(RESULTS_DIR, "pareto_knee_candidates.csv")

OUT_TOP10 = os.path.join(RESULTS_DIR, "pareto_table_top10.csv")
OUT_KNEE = os.path.join(RESULTS_DIR, "pareto_table_knee.csv")


KEEP_COLS = [
    "n_assets",
    "total_cost_usd",
    "total_emissions_kgco2",
    "avg_recycling_eff",
    "total_tokens",
    "avg_latency_s",
    "avg_gas_cost_usd",
    "theta",
    "rho_min",
    "lambda_token",
    "onchain_ratio",
    "use_zkp",
]


def main() -> None:
    feas = pd.read_csv(FEAS_CSV)
    knee = pd.read_csv(KNEE_CSV)

    # Keep only relevant columns
    feas = feas[KEEP_COLS + ["emission_limit", "recycling_min"]].copy()

    # Diversity by constraint-pair, cheapest per pair
    feas = feas.sort_values("total_cost_usd", ascending=True)
    diverse = feas.drop_duplicates(subset=["emission_limit", "recycling_min"], keep="first")

    # Spread 10 points across the diverse set
    if len(diverse) >= 10:
        idx = np.linspace(0, len(diverse) - 1, 10).round().astype(int)
        top10 = diverse.iloc[idx].copy()
    else:
        top10 = feas.head(10).copy()

    # Formatting
    for c in ["total_cost_usd", "total_emissions_kgco2", "total_tokens", "emission_limit"]:
        top10[c] = top10[c].round(3)
    top10["avg_recycling_eff"] = top10["avg_recycling_eff"].round(4)
    top10["avg_latency_s"] = top10["avg_latency_s"].round(3)
    top10["avg_gas_cost_usd"] = top10["avg_gas_cost_usd"].round(4)
    top10["recycling_min"] = top10["recycling_min"].round(3)

    top10.to_csv(OUT_TOP10, index=False)

    # Knee: keep the first row (best)
    knee_point = knee.iloc[[0]].copy()
    keep = [c for c in (KEEP_COLS + ["knee_score", "emission_limit", "recycling_min"]) if c in knee_point.columns]
    knee_point = knee_point[keep]
    knee_point.to_csv(OUT_KNEE, index=False)

    print(f"[OK] Wrote: {OUT_TOP10} (n={len(top10)})")
    print(f"[OK] Wrote: {OUT_KNEE}")
    print("\nTop10 preview:")
    print(top10.to_string(index=False))
    print("\nKnee point:")
    print(pd.read_csv(OUT_KNEE).to_string(index=False))


if __name__ == "__main__":
    main()
