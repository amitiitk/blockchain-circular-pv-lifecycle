"""
plot_pareto_scatter_journal.py

Journal-ready Pareto scatter:
- Cost vs Emissions
- Color: Recycling efficiency
- Highlights knee solution (from results/pareto_table_knee.csv if available; else best knee candidate)
Outputs:
- results/figures/fig_pareto_scatter_JOURNAL.png
- results/figures/fig_pareto_scatter_JOURNAL.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_FEAS = os.path.join(RESULTS, "pareto_frontier_feasible.csv")
IN_KNEE = os.path.join(RESULTS, "pareto_table_knee.csv")
IN_KNEE_CAND = os.path.join(RESULTS, "pareto_knee_candidates.csv")

OUT_PNG = os.path.join(FIG_DIR, "fig_pareto_scatter_JOURNAL.png")
OUT_PDF = os.path.join(FIG_DIR, "fig_pareto_scatter_JOURNAL.pdf")

df = pd.read_csv(IN_FEAS).dropna(subset=["total_cost_usd", "total_emissions_kgco2", "avg_recycling_eff"]).copy()

# Try knee point (preferred)
knee = None
if os.path.exists(IN_KNEE):
    kdf = pd.read_csv(IN_KNEE)
    if len(kdf) >= 1 and {"total_cost_usd", "total_emissions_kgco2"}.issubset(kdf.columns):
        knee = kdf.iloc[0].to_dict()

# Fallback to best candidate
if knee is None and os.path.exists(IN_KNEE_CAND):
    kc = pd.read_csv(IN_KNEE_CAND).dropna(subset=["knee_score"]).copy()
    if len(kc) > 0:
        kc = kc.sort_values("knee_score", ascending=False)
        knee = kc.iloc[0].to_dict()

x = df["total_emissions_kgco2"].values
y = df["total_cost_usd"].values
c = df["avg_recycling_eff"].values

plt.figure(figsize=(7.6, 5.0))
ax = plt.gca()

# Slightly larger markers, edge to improve print clarity
sc = ax.scatter(x, y, c=c, s=55, alpha=0.9, linewidths=0.4)

cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Avg. recycling efficiency")

ax.set_xlabel("Total emissions (kgCO2)")
ax.set_ylabel("Total cost (USD)")
ax.set_title("Feasible Pareto frontier (ε-constraint)")

# Add subtle grid for journal readability
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

# Make axes a bit tighter with padding
xpad = (x.max() - x.min()) * 0.08 if x.max() > x.min() else 1.0
ypad = (y.max() - y.min()) * 0.10 if y.max() > y.min() else 1.0
ax.set_xlim(x.min() - xpad, x.max() + xpad)
ax.set_ylim(y.min() - ypad, y.max() + ypad)

# Highlight knee point
if knee is not None:
    kx = float(knee["total_emissions_kgco2"])
    ky = float(knee["total_cost_usd"])
    ax.scatter([kx], [ky], s=180, marker="*", linewidths=0.8, zorder=5)
    ax.annotate(
        "Knee solution",
        xy=(kx, ky),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
    )

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=400)
plt.savefig(OUT_PDF)
print(f"[OK] Wrote: {OUT_PNG}")
print(f"[OK] Wrote: {OUT_PDF}")
