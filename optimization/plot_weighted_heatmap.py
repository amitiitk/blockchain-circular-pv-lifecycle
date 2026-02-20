"""
plot_weighted_heatmap.py

Figure: Heatmap over (alpha, beta) showing best scalarized score (rank=1).
Input:
- results/weighted_frontier.csv  (from weighted_sum_frontier.py)

Outputs:
- results/figures/fig_weighted_heatmap.png
- results/figures/fig_weighted_heatmap.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_CSV = os.path.join(RESULTS, "weighted_frontier.csv")
OUT_PNG = os.path.join(FIG_DIR, "fig_weighted_heatmap.png")
OUT_PDF = os.path.join(FIG_DIR, "fig_weighted_heatmap.pdf")

df = pd.read_csv(IN_CSV)

# Keep only rank=1 (best per alpha,beta)
df = df[df["rank"] == 1].copy()

alphas = sorted(df["alpha"].unique())
betas = sorted(df["beta"].unique())

# Create matrix of score
M = np.full((len(betas), len(alphas)), np.nan, dtype=float)
for i, b in enumerate(betas):
    for j, a in enumerate(alphas):
        sub = df[(df["alpha"] == a) & (df["beta"] == b)]
        if len(sub) > 0:
            M[i, j] = float(sub["score"].iloc[0])

plt.figure(figsize=(7.2, 4.8))
im = plt.imshow(M, aspect="auto", origin="lower")
cb = plt.colorbar(im)
cb.set_label("Best scalarized score (lower is better)")

plt.xticks(range(len(alphas)), [str(a) for a in alphas], rotation=45, ha="right")
plt.yticks(range(len(betas)), [str(b) for b in betas])

plt.xlabel("alpha (emissions weight)")
plt.ylabel("beta (recycling weight)")
plt.title("Weighted-sum frontier: best score surface")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
print(f"[OK] Wrote: {OUT_PNG}")
print(f"[OK] Wrote: {OUT_PDF}")
