"""
weighted_sum_frontier.py

Purpose
-------
Generate a diverse multiobjective frontier using weighted-sum scalarization.

Why
---
If ε-constraint (cost-min subject to eps) collapses to very few unique solutions,
weighted sums across many (alpha, beta) weights can yield richer diversity.

We solve (on normalized objectives):
  minimize  cost_norm + alpha * emissions_norm + beta * (1 - recycling_norm)

Key enhancement (to avoid collapse):
-----------------------------------
Instead of selecting only the single best solution per (alpha, beta),
we collect TOP_K_PER_WEIGHT best candidates per weight pair, then deduplicate
by decision variables. This almost always increases the number of unique
solutions available for manuscript tables.

Outputs
-------
- results/weighted_frontier.csv          (all candidates across weights, incl. rank)
- results/weighted_top10_unique.csv      (top 10 UNIQUE decision solutions)

Run
---
python optimization/weighted_sum_frontier.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_ALL = os.path.join(RESULTS_DIR, "weighted_frontier.csv")
OUT_TOP10 = os.path.join(RESULTS_DIR, "weighted_top10_unique.csv")

SEED = 42
rng = np.random.default_rng(SEED)

# Collect more than 1 candidate per (alpha, beta) to enrich diversity
TOP_K_PER_WEIGHT = 10


@dataclass(frozen=True)
class DecisionVars:
    theta: float
    rho_min: float
    lambda_token: float
    onchain_ratio: float
    use_zkp: int


@dataclass
class Outcomes:
    n_assets: int
    total_cost_usd: float
    total_emissions_kgco2: float
    avg_recycling_eff: float
    total_tokens: float
    avg_latency_s: float
    avg_gas_cost_usd: float
    zkp_prove_time_s: float
    zkp_verify_time_ms: float


def simulate_system(d: DecisionVars, n_assets: int, scenario: Dict[str, float]) -> Outcomes:
    """
    Standalone surrogate simulator aligned with epsilon_constraint_solver.py.
    """

    # Maintenance economics
    c_pm = scenario.get("c_pm", 8.0)
    c_cm = scenario.get("c_cm", 40.0)
    downtime_cost = scenario.get("downtime_cost", 5.0)

    # Carbon factors
    emission_factor = scenario.get("emission_factor", 4.0)
    panel_mass_kg = scenario.get("panel_mass_kg", 19.5)

    # Blockchain cost model (tuned to create cost-vs-audit trade-offs)
    gas_usd_base = scenario.get("gas_usd_base", 0.60)
    gas_usd_zkp_overhead = scenario.get("gas_usd_zkp_overhead", 0.20)

    # Latency (seconds)
    latency_base = scenario.get("latency_base", 0.35)
    latency_zkp_overhead = scenario.get("latency_zkp_overhead", 0.20)

    # ZKP timings
    zkp_prove_base = scenario.get("zkp_prove_base", 1.2)
    zkp_verify_base = scenario.get("zkp_verify_base", 18.0)

    # Token / incentive economics
    token_cost_per_token = scenario.get("token_cost_per_token", 0.001)
    audit_penalty_cost_param = scenario.get("audit_penalty_cost", 10.0)

    # ---- Behavior model ----
    pm_action_rate = float(np.clip(0.40 - 0.45 * d.theta, 0.02, 0.35))
    fn_rate = float(np.clip(0.04 + 0.60 * (d.theta - 0.55), 0.02, 0.40))
    cm_event_rate = fn_rate

    trust_boost = 0.02 * d.onchain_ratio + (0.03 if d.use_zkp else 0.0)
    incentive_boost = 0.06 * np.tanh(d.lambda_token / 6.0)
    base_recycling = 0.68
    avg_recycling_eff = float(np.clip(base_recycling + trust_boost + incentive_boost, 0.55, 0.95))

    recycling_gap = max(0.0, d.rho_min - avg_recycling_eff)
    recycling_penalty_cost = 12.0 * recycling_gap  # USD per asset

    eligible_eff = max(0.0, avg_recycling_eff - d.rho_min)
    tokens_per_asset = (d.lambda_token * 1000.0) * eligible_eff
    total_tokens = tokens_per_asset * n_assets

    # ---- Costs ----
    pm_cost = pm_action_rate * c_pm
    cm_cost = cm_event_rate * c_cm

    downtime_hours = float(np.clip(1.5 * cm_event_rate - 0.4 * pm_action_rate, 0.0, 1.0))
    downtime_total_cost = downtime_hours * downtime_cost

    # Blockchain gas cost per asset
    events_per_asset = 3.0
    gas_cost = events_per_asset * d.onchain_ratio * gas_usd_base
    if d.use_zkp:
        gas_cost += events_per_asset * d.onchain_ratio * gas_usd_zkp_overhead

    token_liability_cost = total_tokens * token_cost_per_token / n_assets  # per asset

    # Audit risk penalty (expected disputes/rework) rises when on-chain anchoring is low
    audit_risk = (1.0 - d.onchain_ratio) * (0.10 if d.use_zkp else 0.18)
    audit_penalty_cost = audit_penalty_cost_param * audit_risk  # USD per asset expected

    total_cost_per_asset = (
        pm_cost
        + cm_cost
        + downtime_total_cost
        + recycling_penalty_cost
        + gas_cost
        + token_liability_cost
        + audit_penalty_cost
    )
    total_cost_usd = total_cost_per_asset * n_assets

    # ---- Emissions ----
    processed_mass = panel_mass_kg * n_assets
    gross_emissions = emission_factor * processed_mass
    avoided_emissions = (avg_recycling_eff * processed_mass) * scenario.get("avoided_factor", 1.0)

    chain_overhead = processed_mass * (0.01 * d.onchain_ratio) + (0.005 * processed_mass if d.use_zkp else 0.0)
    audit_emissions_overhead = processed_mass * scenario.get("audit_emissions_factor", 0.002) * audit_risk

    total_emissions_kgco2 = float(
        np.clip(gross_emissions - avoided_emissions + chain_overhead + audit_emissions_overhead, 0.0, None)
    )

    # ---- Latency ----
    avg_latency_s = latency_base + d.onchain_ratio * 0.25 + (latency_zkp_overhead if d.use_zkp else 0.0)
    avg_gas_cost_usd = gas_cost

    zkp_prove_time_s = zkp_prove_base if d.use_zkp else 0.0
    zkp_verify_time_ms = zkp_verify_base if d.use_zkp else 0.0

    return Outcomes(
        n_assets=n_assets,
        total_cost_usd=float(total_cost_usd),
        total_emissions_kgco2=float(total_emissions_kgco2),
        avg_recycling_eff=float(avg_recycling_eff),
        total_tokens=float(total_tokens),
        avg_latency_s=float(avg_latency_s),
        avg_gas_cost_usd=float(avg_gas_cost_usd),
        zkp_prove_time_s=float(zkp_prove_time_s),
        zkp_verify_time_ms=float(zkp_verify_time_ms),
    )


def build_decision_grid() -> List[DecisionVars]:
    theta_grid = np.round(np.linspace(0.55, 0.90, 8), 2)
    rho_grid = np.round(np.linspace(0.65, 0.90, 6), 2)
    lambda_grid = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    onchain_grid = np.array([0.10, 0.20, 0.40, 0.60, 0.80, 1.00])
    zkp_grid = np.array([0, 1], dtype=int)

    grid: List[DecisionVars] = []
    for th in theta_grid:
        for rho in rho_grid:
            for lam in lambda_grid:
                for oc in onchain_grid:
                    for z in zkp_grid:
                        grid.append(DecisionVars(float(th), float(rho), float(lam), float(oc), int(z)))
    return grid


def normalize(arr: np.ndarray) -> np.ndarray:
    amin, amax = float(np.min(arr)), float(np.max(arr))
    if amax == amin:
        return np.zeros_like(arr, dtype=float)
    return (arr - amin) / (amax - amin)


def main() -> None:
    n_assets = 1000

    # Scenario (tuned to balance gas vs audit so onchain_ratio is not always 1.0)
    scenario = {
        "c_pm": 8.0,
        "c_cm": 40.0,
        "downtime_cost": 5.0,
        "emission_factor": 4.0,
        "panel_mass_kg": 19.5,
        "avoided_factor": 1.0,
        "gas_usd_base": 0.60,
        "gas_usd_zkp_overhead": 0.20,
        "latency_base": 0.35,
        "latency_zkp_overhead": 0.20,
        "zkp_prove_base": 1.2,
        "zkp_verify_base": 18.0,
        "token_cost_per_token": 0.001,
        "audit_penalty_cost": 10.0,
        "audit_emissions_factor": 0.002,
    }

    grid = build_decision_grid()

    # Precompute outcomes for all decisions
    outs: List[Outcomes] = []
    for d in grid:
        outs.append(simulate_system(d, n_assets=n_assets, scenario=scenario))

    cost = np.array([o.total_cost_usd for o in outs], dtype=float)
    emis = np.array([o.total_emissions_kgco2 for o in outs], dtype=float)
    rec = np.array([o.avg_recycling_eff for o in outs], dtype=float)

    cost_n = normalize(cost)
    emis_n = normalize(emis)
    rec_n = normalize(1.0 - rec)  # minimize (1 - recycling)

    # Wider weight grid to explore trade space
    alphas = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=float)
    betas = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=float)

    rows = []
    for a in alphas:
        for b in betas:
            score = cost_n + a * emis_n + b * rec_n

            # Take TOP-K solutions per (alpha, beta) to build a richer candidate set
            topk_idx = np.argsort(score)[:TOP_K_PER_WEIGHT]

            for rank, idx in enumerate(topk_idx, start=1):
                d = grid[int(idx)]
                o = outs[int(idx)]
                rows.append({
                    "alpha": float(a),
                    "beta": float(b),
                    "rank": int(rank),
                    "score": float(score[int(idx)]),

                    "n_assets": n_assets,
                    "total_cost_usd": o.total_cost_usd,
                    "total_emissions_kgco2": o.total_emissions_kgco2,
                    "avg_recycling_eff": o.avg_recycling_eff,
                    "total_tokens": o.total_tokens,
                    "avg_latency_s": o.avg_latency_s,
                    "avg_gas_cost_usd": o.avg_gas_cost_usd,
                    "zkp_prove_time_s": o.zkp_prove_time_s,
                    "zkp_verify_time_ms": o.zkp_verify_time_ms,

                    "theta": d.theta,
                    "rho_min": d.rho_min,
                    "lambda_token": d.lambda_token,
                    "onchain_ratio": d.onchain_ratio,
                    "use_zkp": d.use_zkp,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ALL, index=False)
    print(f"[OK] Wrote: {OUT_ALL} (rows={len(df)})")

    # Top10 unique decisions (deduplicate by decision variables)
    unique = df.drop_duplicates(subset=["theta", "rho_min", "lambda_token", "onchain_ratio", "use_zkp"]).copy()

    # Sort so we keep best trade-offs first
    unique = unique.sort_values(["score", "total_cost_usd", "total_emissions_kgco2"], ascending=[True, True, True])

    top10 = unique.head(10).copy()
    top10.to_csv(OUT_TOP10, index=False)
    print(f"[OK] Wrote: {OUT_TOP10} (unique_rows={len(top10)})")

    print("\nTop10 UNIQUE preview:")
    print(
        top10[
            [
                "alpha",
                "beta",
                "rank",
                "total_cost_usd",
                "total_emissions_kgco2",
                "avg_recycling_eff",
                "theta",
                "rho_min",
                "lambda_token",
                "onchain_ratio",
                "use_zkp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
