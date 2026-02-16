"""
epsilon_constraint_solver.py

Revised version with relaxed emission epsilon grid
to ensure meaningful Pareto feasibility for IJPE submission.

Generates:
    results/pareto_frontier.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SEED = 42
rng = np.random.default_rng(SEED)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "pareto_frontier.csv")


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

    c_pm = scenario.get("c_pm", 8.0)
    c_cm = scenario.get("c_cm", 40.0)
    downtime_cost = scenario.get("downtime_cost", 5.0)

    emission_factor = scenario.get("emission_factor", 4.0)
    panel_mass_kg = scenario.get("panel_mass_kg", 19.5)
    avoided_factor = scenario.get("avoided_factor", 1.0)

    gas_usd_base = scenario.get("gas_usd_base", 0.06)
    gas_usd_zkp_overhead = scenario.get("gas_usd_zkp_overhead", 0.02)

    latency_base = scenario.get("latency_base", 0.35)
    latency_zkp_overhead = scenario.get("latency_zkp_overhead", 0.20)

    zkp_prove_base = scenario.get("zkp_prove_base", 1.2)
    zkp_verify_base = scenario.get("zkp_verify_base", 18.0)

    token_cost_per_token = scenario.get("token_cost_per_token", 0.0001)

    pm_action_rate = float(np.clip(0.40 - 0.45 * d.theta, 0.02, 0.35))
    fn_rate = float(np.clip(0.05 + 0.35 * (d.theta - 0.6), 0.02, 0.30))
    cm_event_rate = fn_rate

    trust_boost = 0.02 * d.onchain_ratio + (0.03 if d.use_zkp else 0.0)
    incentive_boost = 0.06 * np.tanh(d.lambda_token / 6.0)
    base_recycling = 0.68
    avg_recycling_eff = float(np.clip(base_recycling + trust_boost + incentive_boost, 0.55, 0.95))

    recycling_gap = max(0.0, d.rho_min - avg_recycling_eff)
    recycling_penalty_cost = 12.0 * recycling_gap

    eligible_eff = max(0.0, avg_recycling_eff - d.rho_min)
    tokens_per_asset = (d.lambda_token * 1000.0) * eligible_eff
    total_tokens = tokens_per_asset * n_assets

    pm_cost = pm_action_rate * c_pm
    cm_cost = cm_event_rate * c_cm

    downtime_hours = float(np.clip(1.5 * cm_event_rate - 0.4 * pm_action_rate, 0.0, 1.0))
    downtime_total_cost = downtime_hours * downtime_cost

    events_per_asset = 3.0
    gas_cost = events_per_asset * d.onchain_ratio * gas_usd_base
    if d.use_zkp:
        gas_cost += events_per_asset * d.onchain_ratio * gas_usd_zkp_overhead

    token_liability_cost = total_tokens * token_cost_per_token / n_assets

    total_cost_per_asset = (
        pm_cost
        + cm_cost
        + downtime_total_cost
        + recycling_penalty_cost
        + gas_cost
        + token_liability_cost
    )
    total_cost_usd = total_cost_per_asset * n_assets

    processed_mass = panel_mass_kg * n_assets
    gross_emissions = emission_factor * processed_mass
    avoided_emissions = (avg_recycling_eff * processed_mass) * avoided_factor
    chain_overhead = processed_mass * (0.01 * d.onchain_ratio) + (0.005 * processed_mass if d.use_zkp else 0.0)

    total_emissions_kgco2 = float(np.clip(gross_emissions - avoided_emissions + chain_overhead, 0.0, None))

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
    theta_grid = np.round(np.linspace(0.60, 0.90, 7), 2)
    rho_grid = np.round(np.linspace(0.60, 0.90, 7), 2)
    lambda_grid = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    onchain_grid = np.array([0.20, 0.40, 0.60, 0.80, 1.00])
    zkp_grid = np.array([0, 1], dtype=int)

    grid = []
    for th in theta_grid:
        for rho in rho_grid:
            for lam in lambda_grid:
                for oc in onchain_grid:
                    for z in zkp_grid:
                        grid.append(DecisionVars(th, rho, lam, oc, z))
    return grid


def epsilon_constraint_pareto(n_assets, emission_epsilons, recycling_epsilons, scenario):

    grid = build_decision_grid()
    rows = []

    for e_lim in emission_epsilons:
        for r_min in recycling_epsilons:

            best = None

            for d in grid:
                out = simulate_system(d, n_assets, scenario)

                if out.total_emissions_kgco2 <= e_lim and out.avg_recycling_eff >= r_min:
                    if best is None or out.total_cost_usd < best[0]:
                        best = (out.total_cost_usd, out, d)

            if best is None:
                rows.append({
                    "n_assets": n_assets,
                    "emission_limit": e_lim,
                    "recycling_min": r_min,
                    "feasible": 0
                })
            else:
                _, out, d = best
                rows.append({
                    "n_assets": n_assets,
                    "emission_limit": e_lim,
                    "recycling_min": r_min,
                    "feasible": 1,
                    "total_cost_usd": out.total_cost_usd,
                    "total_emissions_kgco2": out.total_emissions_kgco2,
                    "avg_recycling_eff": out.avg_recycling_eff,
                    "total_tokens": out.total_tokens,
                    "avg_latency_s": out.avg_latency_s,
                    "avg_gas_cost_usd": out.avg_gas_cost_usd,
                    "theta": d.theta,
                    "rho_min": d.rho_min,
                    "lambda_token": d.lambda_token,
                    "onchain_ratio": d.onchain_ratio,
                    "use_zkp": d.use_zkp,
                })

    return pd.DataFrame(rows)


def main():

    n_assets = 1000

    scenario = {
        "c_pm": 8.0,
        "c_cm": 40.0,
        "downtime_cost": 5.0,
        "emission_factor": 4.0,
        "panel_mass_kg": 19.5,
        "avoided_factor": 1.0,
        "gas_usd_base": 0.06,
        "gas_usd_zkp_overhead": 0.02,
        "latency_base": 0.35,
        "latency_zkp_overhead": 0.20,
        "zkp_prove_base": 1.2,
        "zkp_verify_base": 18.0,
        "token_cost_per_token": 0.0001,
    }

    ref_d = DecisionVars(0.70, 0.75, 6.0, 0.60, 1)
    ref_out = simulate_system(ref_d, n_assets, scenario)

    # 🔹 RELAXED EMISSION EPSILON GRID
    emission_epsilons = list(np.linspace(
        1.00 * ref_out.total_emissions_kgco2,
        1.70 * ref_out.total_emissions_kgco2,
        9
    ))

    recycling_epsilons = list(np.linspace(0.65, 0.90, 6))

    df = epsilon_constraint_pareto(n_assets, emission_epsilons, recycling_epsilons, scenario)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote Pareto frontier to: {OUTPUT_CSV}")
    print(df.head(12))


if __name__ == "__main__":
    main()
