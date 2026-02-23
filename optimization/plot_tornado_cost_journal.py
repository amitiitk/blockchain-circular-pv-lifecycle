"""
plot_tornado_cost_journal.py

Journal-ready tornado plot:
- Uses results/sensitivity_summary.csv
- Shows TOP_N parameters by max absolute cost impact
- Symmetric x-limits around 0
Outputs:
- results/figures/fig_tornado_cost_JOURNAL.png
- results/figures/fig_tornado_cost_JOURNAL.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_CSV = os.path.join(RESULTS, "sensitivity_summary.csv")
OUT_PNG = os.path.join(FIG_DIR, "fig_tornado_cost_JOURNAL.png")
OUT_PDF = os.path.join(FIG_DIR, "fig_tornado_cost_JOURNAL.pdf")

TOP_N = 8  # journal-friendly (6–10 works well)

df = pd.read_csv(IN_CSV)
df = df[df["metric"] == "pct_total_cost_usd"].copy()

# compute bounds per parameter
df["abs_change"] = df["pct_change"].abs()
rank = df.groupby("parameter")["abs_change"].max().sort_values(ascending=False).head(TOP_N)

bounds = df.groupby("parameter")["pct_change"].agg(["min", "max"]).reindex(rank.index)

# Reverse for plotting top at top
bounds = bounds.iloc[::-1]
params = bounds.index.tolist()
mins = bounds["min"].values
maxs = bounds["max"].values

plt.figure(figsize=(7.6, 5.2))
ax = plt.gca()

ypos = np.arange(len(params))

# Draw thick horizontal bars
ax.hlines(y=ypos, xmin=mins, xmax=maxs, linewidth=12, alpha=0.9)
ax.axvline(0, linewidth=1.0)

ax.set_yticks(ypos)
ax.set_yticklabels(params)
ax.set_xlabel("% change in total cost")
ax.set_title("One-way sensitivity (Total cost): top drivers")

# Symmetric x-limit around 0 for tornado readability
lim = float(max(np.max(np.abs(mins)), np.max(np.abs(maxs))))
ax.set_xlim(-1.15 * lim, 1.15 * lim)

ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=400)
plt.savefig(OUT_PDF)
print(f"[OK] Wrote: {OUT_PNG}")
print(f"[OK] Wrote: {OUT_PDF}")
