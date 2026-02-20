"""
plot_tornado_cost.py

Figure: Tornado-style bar chart for cost sensitivity.
Input:
- results/sensitivity_summary.csv  (from sensitivity_summary.py)

Outputs:
- results/figures/fig_tornado_cost.png
- results/figures/fig_tornado_cost.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_CSV = os.path.join(RESULTS, "sensitivity_summary.csv")
OUT_PNG = os.path.join(FIG_DIR, "fig_tornado_cost.png")
OUT_PDF = os.path.join(FIG_DIR, "fig_tornado_cost.pdf")

df = pd.read_csv(IN_CSV)

# Filter to cost metric only
df = df[df["metric"] == "pct_total_cost_usd"].copy()

# We want max absolute change per parameter for tornado ranking
df["abs_change"] = df["pct_change"].abs()
rank = df.groupby("parameter")["abs_change"].max().sort_values(ascending=True)

# For each parameter, get the min and max pct_change
bounds = df.groupby("parameter")["pct_change"].agg(["min", "max"]).reindex(rank.index)

params = bounds.index.tolist()
mins = bounds["min"].values
maxs = bounds["max"].values

plt.figure(figsize=(7.2, 5.2))
ypos = range(len(params))

# Draw bars from min to max (tornado)
plt.hlines(y=ypos, xmin=mins, xmax=maxs, linewidth=10)
plt.axvline(0, linewidth=1)

plt.yticks(list(ypos), params)
plt.xlabel("% change in total cost")
plt.title("One-way sensitivity (Total cost)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
print(f"[OK] Wrote: {OUT_PNG}")
print(f"[OK] Wrote: {OUT_PDF}")
