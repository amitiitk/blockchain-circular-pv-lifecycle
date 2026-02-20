"""
sensitivity_summary.py

Reads results/sensitivity_results.csv and produces:
- results/sensitivity_summary.csv

Summaries:
- Top parameter impacts by absolute % change for:
  * total_cost_usd
  * total_emissions_kgco2
  * total_tokens
  * avg_latency_s
  * avg_gas_cost_usd

Run:
python optimization/sensitivity_summary.py
"""

import os
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

IN_CSV = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "sensitivity_summary.csv")


def top_impacts(df: pd.DataFrame, metric_col: str, top_k: int = 8) -> pd.DataFrame:
    tmp = df.copy()
    tmp["abs_impact"] = tmp[metric_col].abs()
    tmp = tmp.sort_values("abs_impact", ascending=False).head(top_k)
    return tmp[["parameter", "multiplier", "baseline_value", "new_value", metric_col]].rename(
        columns={metric_col: "pct_change"}
    )


def main() -> None:
    df = pd.read_csv(IN_CSV)

    blocks = []
    for metric in [
        "pct_total_cost_usd",
        "pct_total_emissions_kgco2",
        "pct_total_tokens",
        "pct_avg_latency_s",
        "pct_avg_gas_cost_usd",
    ]:
        top = top_impacts(df, metric_col=metric, top_k=8)
        top.insert(0, "metric", metric)
        blocks.append(top)

    out = pd.concat(blocks, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote: {OUT_CSV}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
