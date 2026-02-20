"""
plot_pareto_scatter.py

Figure: Pareto scatter (Cost vs Emissions), colored by Recycling efficiency.
Inputs:
- results/pareto_frontier_feasible.csv  (from pareto_postprocess.py)

Outputs:
- results/figures/fig_pareto_scatter.png
- results/figures/fig_pareto_scatter.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_CSV = os.path.join(RESULTS, "pareto_frontier_feasible.csv")
OUT_PNG = os.path.join(FIG_DIR, "fig_pareto_scatter.png")
OUT_PDF = os.path.join(FIG_DIR, "fig_pareto_scatter.pdf")

df = pd.read_csv(IN_CSV)

# Basic cleanup
df = df.dropna(subset=["total_cost_usd", "total_emissions_kgco2", "avg_recycling_eff"]).copy()

x = df["total_emissions_kgco2"].values
y = df["total_cost_usd"].values
c = df["avg_recycling_eff"].values

plt.figure(figsize=(7.2, 4.8))
sc = plt.scatter(x, y, c=c, s=35)
cb = plt.colorbar(sc)
cb.set_label("Avg. recycling efficiency")

plt.xlabel("Total emissions (kgCO2)")
plt.ylabel("Total cost (USD)")
plt.title("Feasible Pareto frontier (ε-constraint)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
print(f"[OK] Wrote: {OUT_PNG}")
print(f"[OK] Wrote: {OUT_PDF}")
