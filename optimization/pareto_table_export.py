"""
pareto_table_export.py

Creates manuscript-ready Pareto tables from results/pareto_frontier_feasible.csv:

Outputs:
- results/pareto_table_top10.csv          (10 feasible points sampled across the frontier)
- results/pareto_table_top10_unique.csv   (10 UNIQUE decision solutions)
- results/pareto_table_knee.csv           (single knee point)

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
OUT_TOP10_UNIQUE = os.path.join(RESULTS_DIR, "pareto_table_top10_unique.csv")
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


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["total_cost_usd", "total_emissions_kgco2", "total_tokens", "emission_limit"]:
        if c in df.columns:
            df[c] = df[c].round(3)
    if "avg_recycling_eff" in df.columns:
        df["avg_recycling_eff"] = df["avg_recycling_eff"].round(4)
    if "avg_latency_s" in df.columns:
        df["avg_latency_s"] = df["avg_latency_s"].round(3)
    if "avg_gas_cost_usd" in df.columns:
        df["avg_gas_cost_usd"] = df["avg_gas_cost_usd"].round(4)
    if "recycling_min" in df.columns:
        df["recycling_min"] = df["recycling_min"].round(3)
    return df


def sample_spread(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Sample n rows evenly across a DataFrame (assumes df is already sorted)."""
    if len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n).round().astype(int)
    return df.iloc[idx].copy()


def main() -> None:
    feas = pd.read_csv(FEAS_CSV)
    knee = pd.read_csv(KNEE_CSV)

    # Keep only relevant columns + constraints
    feas = feas[KEEP_COLS + ["emission_limit", "recycling_min"]].copy()

    # Sort by cost first (common IJPE reporting preference)
    feas = feas.sort_values("total_cost_usd", ascending=True)

    # -----------------------------
    # Table 1: Top10 (feasible spread)
    # -----------------------------
    # Diversity by constraint pairs (cheap representative per epsilon pair)
    diverse_eps = feas.drop_duplicates(subset=["emission_limit", "recycling_min"], keep="first")
    top10 = sample_spread(diverse_eps, n=10)
    top10 = format_table(top10)
    top10.to_csv(OUT_TOP10, index=False)

    # -----------------------------
    # Table 2: Top10 UNIQUE solutions (no duplicate decisions)
    # -----------------------------
    # Diversity by unique decision variables (avoid repeated identical solutions)
    diverse_decisions = feas.drop_duplicates(
        subset=["theta", "rho_min", "lambda_token", "onchain_ratio", "use_zkp"],
        keep="first",
    )

    top10_unique = sample_spread(diverse_decisions, n=10)
    top10_unique = format_table(top10_unique)
    top10_unique.to_csv(OUT_TOP10_UNIQUE, index=False)

    # -----------------------------
    # Table 3: Knee point
    # -----------------------------
    knee_point = knee.iloc[[0]].copy()
    keep = [c for c in (KEEP_COLS + ["knee_score", "emission_limit", "recycling_min"]) if c in knee_point.columns]
    knee_point = knee_point[keep]
    knee_point = format_table(knee_point)
    knee_point.to_csv(OUT_KNEE, index=False)

    print(f"[OK] Wrote: {OUT_TOP10} (n={len(top10)})")
    print(f"[OK] Wrote: {OUT_TOP10_UNIQUE} (n={len(top10_unique)})")
    print(f"[OK] Wrote: {OUT_KNEE}")

    print("\nTop10 UNIQUE preview:")
    print(top10_unique.to_string(index=False))

    print("\nKnee point:")
    print(pd.read_csv(OUT_KNEE).to_string(index=False))


if __name__ == "__main__":
    main()
