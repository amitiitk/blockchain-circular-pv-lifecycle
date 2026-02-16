"""
epsilon_constraint_solver.py

Purpose
-------
Generate a Pareto frontier (ε-constraint method) for the IJPE revision package.

This script is intentionally self-contained and deterministic. It:
- Simulates lifecycle economics + emissions + recycling outcomes
- Searches over a grid of decision variables
- Uses ε-constraints on Emissions and Recycling
- Minimizes Total Cost subject to constraints
- Writes results/pareto_frontier.csv

Notes
-----
This is a *reproducible* scaffold. You can later replace the simulation block with:
- real data
- a MILP/LP solver
- a more detailed carbon/revenue calibration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Deterministic experiment setup
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# Output paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "pareto_frontier.csv")


@dataclass(frozen=True)
class DecisionVars:
    """
    Decision variables you can interpret as:
    - theta: PM failure threshold (higher = fewer PM actions)
    - rho_min: minimum recycling efficiency required for incentives/compliance
    - lambda_token: token multiplier (incentive intensity)
    - onchain_ratio: fraction of records stored on-chain vs off-chain (IPFS)
    - use_zkp: whether compliance uses ZKP (privacy) or not
    """
    theta: float
    rho_min: float
    lambda_token: float
    onchain_ratio: float
    use_zkp: int


@dataclass
class Outcomes:
    """
    System outcomes (per instance).
    Values are normalized per-asset for readability, then scaled by N assets.
    """
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
    A deterministic, explainable surrogate simulator.

    Replace this with your true pipeline later (PM model outputs, gas profiling logs,
    carbon calibration table, commodity prices, etc.). For IJPE revision, this provides:
    - consistent Pareto points
    - sensitivity hooks (scenario dict)
    """

    # ---- Scenario parameters (tunable later) ----
    # Maintenance economics
    c_pm = scenario.get("c_pm", 8.0)          # preventive maintenance cost per action (USD)
    c_cm = scenario.get("c_cm", 40.0)         # corrective maintenance cost (USD)
    downtime_cost = scenario.get("downtime_cost", 5.0)  # USD per downtime hour-equivalent

    # Carbon factors
    emission_factor = scenario.get("emission_factor", 4.0)  # kg CO2 per kg processed (proxy)
    panel_mass_kg = scenario.get("panel_mass_kg", 19.5)     # average panel mass

    # Blockchain cost model
    gas_usd_base = scenario.get("gas_usd_base", 0.06)       # baseline cost per tx (USD) on private net proxy
    gas_usd_zkp_overhead = scenario.get("gas_usd_zkp_overhead", 0.02)

    # Latency (seconds)
    latency_base = scenario.get("latency_base", 0.35)
    latency_zkp_overhead = scenario.get("latency_zkp_overhead", 0.20)

    # ZKP timings
    zkp_prove_base = scenario.get("zkp_prove_base", 1.2)    # seconds
    zkp_verify_base = scenario.get("zkp_verify_base", 18.0) # milliseconds

    # Token / incentive economics
    token_cost_per_token = scenario.get("token_cost_per_token", 0.0001)  # USD per token (proxy for reserve liability)

    # ---- Derived behavior model ----
    # PM action rate decreases as theta increases (higher threshold => fewer alerts)
    # Bounded in [0.02, 0.35] for realism.
    pm_action_rate = float(np.clip(0.40 - 0.45 * d.theta, 0.02, 0.35))

    # False negatives increase with higher theta (missing failures).
    # Bounded in [0.02, 0.30]
    fn_rate = float(np.clip(0.05 + 0.35 * (d.theta - 0.6), 0.02, 0.30))

    # Corrective events depend on missed failures
    cm_event_rate = fn_rate

    # Recycling efficiency is influenced by rho_min (regulatory stringency),
    # token multiplier (incentives), and data quality (on-chain anchoring + ZKP trust).
    trust_boost = 0.02 * d.onchain_ratio + (0.03 if d.use_zkp else 0.0)
    incentive_boost = 0.06 * np.tanh(d.lambda_token / 6.0)
    # Base recycling without incentives
    base_recycling = 0.68

    avg_recycling_eff = float(np.clip(base_recycling + trust_boost + incentive_boost, 0.55, 0.95))

    # Enforce rho_min as compliance / eligibility pressure:
    # if rho_min > achievable recycling, you incur penalty costs (extra sorting/handling)
    recycling_gap = max(0.0, d.rho_min - avg_recycling_eff)
    recycling_penalty_cost = 12.0 * recycling_gap  # USD per asset penalty proxy

    # Token payouts increase with recycling efficiency and lambda_token, but only above rho_min
    eligible_eff = max(0.0, avg_recycling_eff - d.rho_min)
    tokens_per_asset = (d.lambda_token * 1000.0) * eligible_eff  # proxy
    total_tokens = tokens_per_asset * n_assets

    # ---- Costs ----
    # Maintenance cost per asset
    pm_cost = pm_action_rate * c_pm
    cm_cost = cm_event_rate * c_cm

    # Downtime increases with corrective events; PM reduces downtime slightly
    downtime_hours = float(np.clip(1.5 * cm_event_rate - 0.4 * pm_action_rate, 0.0, 1.0))
    downtime_total_cost = downtime_hours * downtime_cost

    # Blockchain cost per asset: events scale with onchain_ratio
    # Assume 3 lifecycle events per asset (register, maint, recycle/audit)
    events_per_asset = 3.0
    gas_cost = events_per_asset * d.onchain_ratio * gas_usd_base
    if d.use_zkp:
        gas_cost += events_per_asset * d.onchain_ratio * gas_usd_zkp_overhead

    # Token reserve / redemption liability cost proxy
    token_liability_cost = total_tokens * token_cost_per_token / n_assets  # per-asset

    total_cost_per_asset = (
        pm_cost
        + cm_cost
        + downtime_total_cost
        + recycling_penalty_cost
        + gas_cost
        + token_liability_cost
    )
    total_cost_usd = total_cost_per_asset * n_assets

    # ---- Emissions ----
    # Emissions decrease with higher recycling efficiency (avoided virgin production proxy),
    # but increase slightly with higher on-chain ratio and ZKP computation overhead.
    processed_mass = panel_mass_kg * n_assets
    gross_emissions = emission_factor * processed_mass  # proxy for processing footprint

    avoided_emissions = (avg_recycling_eff * processed_mass) * scenario.get("avoided_factor", 1.0)
    chain_overhead = processed_mass * (0.01 * d.onchain_ratio) + (0.005 * processed_mass if d.use_zkp else 0.0)

    total_emissions_kgco2 = float(np.clip(gross_emissions - avoided_emissions + chain_overhead, 0.0, None))

    # ---- Latency ----
    avg_latency_s = latency_base + d.onchain_ratio * 0.25 + (latency_zkp_overhead if d.use_zkp else 0.0)
    avg_gas_cost_usd = gas_cost  # per asset gas proxy

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
    """
    Define a compact but meaningful decision grid.
    Increase resolution later for a smoother frontier (at higher compute cost).
    """
    theta_grid = np.round(np.linspace(0.60, 0.90, 7), 2)             # PM threshold
    rho_grid = np.round(np.linspace(0.60, 0.90, 7), 2)               # min recycling efficiency
    lambda_grid = np.array([2.0, 4.0, 6.0, 8.0, 10.0])               # token multiplier
    onchain_grid = np.array([0.20, 0.40, 0.60, 0.80, 1.00])          # on-chain vs IPFS
    zkp_grid = np.array([0, 1], dtype=int)

    grid: List[DecisionVars] = []
    for th in theta_grid:
        for rho in rho_grid:
            for lam in lambda_grid:
                for oc in onchain_grid:
                    for z in zkp_grid:
                        grid.append(DecisionVars(theta=float(th), rho_min=float(rho),
                                                 lambda_token=float(lam), onchain_ratio=float(oc),
                                                 use_zkp=int(z)))
    return grid


def epsilon_constraint_pareto(
    n_assets: int,
    emission_epsilons: List[float],
    recycling_epsilons: List[float],
    scenario: Dict[str, float],
) -> pd.DataFrame:
    """
    For each (emission_limit, recycling_min) pair:
    minimize cost subject to:
      emissions <= emission_limit
      recycling >= recycling_min
    """
    grid = build_decision_grid()

    rows = []
    for e_lim in emission_epsilons:
        for r_min in recycling_epsilons:
            best: Tuple[float, Outcomes, DecisionVars] | None = None

            for d in grid:
                out = simulate_system(d, n_assets=n_assets, scenario=scenario)

                if out.total_emissions_kgco2 <= e_lim and out.avg_recycling_eff >= r_min:
                    if best is None or out.total_cost_usd < best[0]:
                        best = (out.total_cost_usd, out, d)

            if best is None:
                # No feasible solution for this ε-pair
                rows.append({
                    "n_assets": n_assets,
                    "emission_limit": e_lim,
                    "recycling_min": r_min,
                    "feasible": 0,
                    "total_cost_usd": np.nan,
                    "total_emissions_kgco2": np.nan,
                    "avg_recycling_eff": np.nan,
                    "total_tokens": np.nan,
                    "avg_latency_s": np.nan,
                    "avg_gas_cost_usd": np.nan,
                    "zkp_prove_time_s": np.nan,
                    "zkp_verify_time_ms": np.nan,
                    "theta": np.nan,
                    "rho_min": np.nan,
                    "lambda_token": np.nan,
                    "onchain_ratio": np.nan,
                    "use_zkp": np.nan,
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
                    "zkp_prove_time_s": out.zkp_prove_time_s,
                    "zkp_verify_time_ms": out.zkp_verify_time_ms,
                    "theta": d.theta,
                    "rho_min": d.rho_min,
                    "lambda_token": d.lambda_token,
                    "onchain_ratio": d.onchain_ratio,
                    "use_zkp": d.use_zkp,
                })

    return pd.DataFrame(rows)


def main() -> None:
    # Choose instance size(s). Start with 1k for quick iteration; scale later.
    n_assets = 1000

    # Baseline scenario (tweak later, or load from YAML/CSV if you prefer)
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

    # Build ε ranges (based on baseline magnitudes)
    # We'll compute a baseline reference point to set reasonable ε grids.
    ref_d = DecisionVars(theta=0.70, rho_min=0.75, lambda_token=6.0, onchain_ratio=0.60, use_zkp=1)
    ref_out = simulate_system(ref_d, n_assets=n_assets, scenario=scenario)

    # Emission limits: from 70% to 100% of reference emissions (tight to loose)
    emission_epsilons = list(np.linspace(0.70 * ref_out.total_emissions_kgco2, 1.00 * ref_out.total_emissions_kgco2, 7))
    emission_epsilons = [float(round(x, 2)) for x in emission_epsilons]

    # Recycling minimum constraints: from 0.65 to 0.90
    recycling_epsilons = list(np.linspace(0.65, 0.90, 6))
    recycling_epsilons = [float(round(x, 3)) for x in recycling_epsilons]

    df = epsilon_constraint_pareto(
        n_assets=n_assets,
        emission_epsilons=emission_epsilons,
        recycling_epsilons=recycling_epsilons,
        scenario=scenario,
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote Pareto frontier to: {OUTPUT_CSV}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
