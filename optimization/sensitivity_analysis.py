"""
sensitivity_analysis.py

Purpose
-------
One-way sensitivity analysis for the IJPE revision package.

- Uses the same surrogate simulator used in epsilon_constraint_solver.py
- Varies one parameter at a time (±20% by default)
- Reports impacts on Cost / Emissions / Recycling / Tokens / Latency / Gas
- Writes results/sensitivity_results.csv

How to run
----------
python optimization/sensitivity_analysis.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


# -----------------------------
# Deterministic experiment setup
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "sensitivity_results.csv")


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
    Same surrogate simulator logic as epsilon_constraint_solver.py (kept duplicated
    here intentionally for standalone reproducibility).
    """

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

    # Maintenance behavior
    pm_action_rate = float(np.clip(0.40 - 0.45 * d.theta, 0.02, 0.35))
    fn_rate = float(np.clip(0.05 + 0.35 * (d.theta - 0.6), 0.02, 0.30))
    cm_event_rate = fn_rate

    # Recycling behavior
    trust_boost = 0.02 * d.onchain_ratio + (0.03 if d.use_zkp else 0.0)
    incentive_boost = 0.06 * np.tanh(d.lambda_token / 6.0)
    base_recycling = 0.68
    avg_recycling_eff = float(np.clip(base_recycling + trust_boost + incentive_boost, 0.55, 0.95))

    recycling_gap = max(0.0, d.rho_min - avg_recycling_eff)
    recycling_penalty_cost = 12.0 * recycling_gap

    eligible_eff = max(0.0, avg_recycling_eff - d.rho_min)
    tokens_per_asset = (d.lambda_token * 1000.0) * eligible_eff
    total_tokens = tokens_per_asset * n_assets

    # Costs
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

    # Emissions
    processed_mass = panel_mass_kg * n_assets
    gross_emissions = emission_factor * processed_mass
    avoided_emissions = (avg_recycling_eff * processed_mass) * avoided_factor
    chain_overhead = processed_mass * (0.01 * d.onchain_ratio) + (0.005 * processed_mass if d.use_zkp else 0.0)
    total_emissions_kgco2 = float(np.clip(gross_emissions - avoided_emissions + chain_overhead, 0.0, None))

    # Latency
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


def pct_change(new: float, base: float) -> float:
    if base == 0:
        return float("nan")
    return 100.0 * (new - base) / base


def main() -> None:
    # Baseline instance
    n_assets = 1000

    # Fixed decision variables (you can later set these to the "knee" Pareto solution)
    d0 = DecisionVars(theta=0.55, rho_min=0.75, lambda_token=8.0, onchain_ratio=0.10, use_zkp=1)

    # Baseline scenario parameters
    base_scenario = {
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

    base_out = simulate_system(d0, n_assets=n_assets, scenario=base_scenario)

    # Sensitivity definition: parameter -> (low_multiplier, high_multiplier)
    # IJPE-friendly: show ±20% around baseline, and a few wider ranges on key factors.
    sens: Dict[str, List[float]] = {
        "emission_factor": [0.85, 1.15],         # e.g., 3.4 to 4.6 if baseline 4.0
        "gas_usd_base": [0.50, 1.50],            # gas volatility
        "gas_usd_zkp_overhead": [0.50, 1.50],
        "downtime_cost": [0.80, 1.20],
        "c_pm": [0.80, 1.20],
        "c_cm": [0.80, 1.20],
        "panel_mass_kg": [0.90, 1.10],
        "avoided_factor": [0.80, 1.20],
        "token_cost_per_token": [0.50, 2.00],    # reserve liability sensitivity
        "latency_base": [0.80, 1.20],
        "latency_zkp_overhead": [0.80, 1.20],
        "zkp_prove_base": [0.80, 1.20],
        "zkp_verify_base": [0.80, 1.20],
    }

    rows = []
    for param, multipliers in sens.items():
        base_val = base_scenario[param]
        for m in multipliers:
            scenario = dict(base_scenario)
            scenario[param] = base_val * m

            out = simulate_system(d0, n_assets=n_assets, scenario=scenario)

            rows.append({
                "parameter": param,
                "multiplier": m,
                "baseline_value": base_val,
                "new_value": scenario[param],

                "base_total_cost_usd": base_out.total_cost_usd,
                "new_total_cost_usd": out.total_cost_usd,
                "pct_total_cost_usd": pct_change(out.total_cost_usd, base_out.total_cost_usd),

                "base_total_emissions_kgco2": base_out.total_emissions_kgco2,
                "new_total_emissions_kgco2": out.total_emissions_kgco2,
                "pct_total_emissions_kgco2": pct_change(out.total_emissions_kgco2, base_out.total_emissions_kgco2),

                "base_avg_recycling_eff": base_out.avg_recycling_eff,
                "new_avg_recycling_eff": out.avg_recycling_eff,
                "pct_avg_recycling_eff": pct_change(out.avg_recycling_eff, base_out.avg_recycling_eff),

                "base_total_tokens": base_out.total_tokens,
                "new_total_tokens": out.total_tokens,
                "pct_total_tokens": pct_change(out.total_tokens, base_out.total_tokens),

                "base_avg_latency_s": base_out.avg_latency_s,
                "new_avg_latency_s": out.avg_latency_s,
                "pct_avg_latency_s": pct_change(out.avg_latency_s, base_out.avg_latency_s),

                "base_avg_gas_cost_usd": base_out.avg_gas_cost_usd,
                "new_avg_gas_cost_usd": out.avg_gas_cost_usd,
                "pct_avg_gas_cost_usd": pct_change(out.avg_gas_cost_usd, base_out.avg_gas_cost_usd),
            })

    df = pd.DataFrame(rows)

    # Sort by absolute impact on total cost (IJPE-like “tornado ordering”)
    df["abs_pct_total_cost_usd"] = df["pct_total_cost_usd"].abs()
    df = df.sort_values(["abs_pct_total_cost_usd", "parameter", "multiplier"], ascending=[False, True, True])
    df = df.drop(columns=["abs_pct_total_cost_usd"])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote sensitivity results to: {OUTPUT_CSV}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
