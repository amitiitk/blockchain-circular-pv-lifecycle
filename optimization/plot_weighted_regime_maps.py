"""
plot_weighted_regime_maps.py

Journal-ready regime maps for weighted frontier:
- Uses results/weighted_frontier.csv (rank=1 per (alpha,beta))
- Heatmap 1: dominant onchain_ratio
- Heatmap 2: dominant lambda_token
- Optional Heatmap 3: dominant use_zkp

Outputs:
- results/figures/fig_weighted_regime_onchain.png/.pdf
- results/figures/fig_weighted_regime_lambda.png/.pdf
- results/figures/fig_weighted_regime_zkp.png/.pdf
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

OUT_ONCHAIN_PNG = os.path.join(FIG_DIR, "fig_weighted_regime_onchain.png")
OUT_ONCHAIN_PDF = os.path.join(FIG_DIR, "fig_weighted_regime_onchain.pdf")

OUT_LAMBDA_PNG = os.path.join(FIG_DIR, "fig_weighted_regime_lambda.png")
OUT_LAMBDA_PDF = os.path.join(FIG_DIR, "fig_weighted_regime_lambda.pdf")

OUT_ZKP_PNG = os.path.join(FIG_DIR, "fig_weighted_regime_zkp.png")
OUT_ZKP_PDF = os.path.join(FIG_DIR, "fig_weighted_regime_zkp.pdf")

df = pd.read_csv(IN_CSV)
df = df[df.get("rank", 1) == 1].copy()  # best per weight
df = df.dropna(subset=["alpha", "beta", "onchain_ratio", "lambda_token", "use_zkp"])

alphas = sorted(df["alpha"].unique())
betas = sorted(df["beta"].unique())

def build_matrix(value_col: str) -> np.ndarray:
    M = np.full((len(betas), len(alphas)), np.nan, dtype=float)
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            sub = df[(df["alpha"] == a) & (df["beta"] == b)]
            if len(sub) > 0:
                M[i, j] = float(sub[value_col].iloc[0])
    return M

def plot_heatmap(M: np.ndarray, title: str, cbar_label: str, out_png: str, out_pdf: str):
    plt.figure(figsize=(7.6, 5.0))
    ax = plt.gca()
    im = ax.imshow(M, aspect="auto", origin="lower")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas], rotation=45, ha="right")
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([str(b) for b in betas])

    ax.set_xlabel("alpha (emissions weight)")
    ax.set_ylabel("beta (recycling weight)")
    ax.set_title(title)

    # gridlines for print clarity
    ax.set_xticks(np.arange(-.5, len(alphas), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(betas), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=400)
    plt.savefig(out_pdf)
    print(f"[OK] Wrote: {out_png}")
    print(f"[OK] Wrote: {out_pdf}")

# 1) On-chain regime map
M_onchain = build_matrix("onchain_ratio")
plot_heatmap(
    M_onchain,
    "Weighted-sum frontier: dominant on-chain anchoring",
    "onchain_ratio",
    OUT_ONCHAIN_PNG,
    OUT_ONCHAIN_PDF,
)

# 2) Lambda regime map
M_lambda = build_matrix("lambda_token")
plot_heatmap(
    M_lambda,
    "Weighted-sum frontier: dominant incentive intensity",
    "lambda_token",
    OUT_LAMBDA_PNG,
    OUT_LAMBDA_PDF,
)

# 3) ZKP regime map (binary)
M_zkp = build_matrix("use_zkp")
plot_heatmap(
    M_zkp,
    "Weighted-sum frontier: dominant ZKP choice",
    "use_zkp (0=no, 1=yes)",
    OUT_ZKP_PNG,
    OUT_ZKP_PDF,
)
