"""Per-run lambda selection for SABRE-MS (the rule described in the paper, Sec.
method-sabrems).

For each lambda in a fixed grid LAMBDA_GRID (which INCLUDES 0, so production
SABRE's score is always a candidate and is selected when no positive lambda
shortens the makespan), route the circuit at a small probe budget K0 and record
the shortest post-cancellation makespan that lambda reaches. Keep the lambda with
the shortest probe makespan, then route once more at the full budget K and return
the shortest-makespan trial.

This is part of the algorithm, not a tuned hyperparameter: nothing outside the
circuit being routed enters the choice. Cost is K0*|grid| + K trials.

select_lambda_ms() returns (best_lambda, best_circuit, best_makespan, probe_table)
where best_circuit is the final K-budget SABRE-MS circuit (post gate-cancellation),
so callers can compute makespan AND ESP on the exact same circuit.
"""
from __future__ import annotations

from exp_real_full import (
    route_with_sabre_ms, optimize_qc, asap_makespan_qc,
)

# Grid INCLUDES 0 (the paper's Lambda = {0, 0.005, ...}); lambda=0 recovers SABRE.
LAMBDA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
K0 = 5    # probe budget per lambda
K = 20    # final budget


def _best_ms_makespan(qc_phys, graph, coupling, lam, seeds, alim_mult):
    """Shortest post-cancellation makespan over the given seeds at this lambda,
    plus the circuit that achieves it (None if every trial failed). Also returns
    the shortest RAW (pre-cancellation) makespan over the same seeds, for the
    scheduling/absorption channel split."""
    best_mk, best_qc = float("inf"), None
    best_raw = float("inf")
    for s in seeds:
        r = route_with_sabre_ms(qc_phys, graph, coupling, lam, seed=s,
                                alim_mult=alim_mult)
        if r is None:
            continue
        raw_mk = asap_makespan_qc(r)
        if raw_mk < best_raw:
            best_raw = raw_mk
        o = optimize_qc(r)
        mk = asap_makespan_qc(o)
        if mk < best_mk:
            best_mk, best_qc = mk, o
    return best_mk, best_qc, best_raw


def select_lambda_ms(qc_phys, graph, coupling, alim_mult=50,
                     grid=LAMBDA_GRID, k0=K0, k=K):
    """Per-run lambda selection + final route. See module docstring.

    Returns dict with keys:
      lam            chosen lambda
      ms_qc          final SABRE-MS circuit (post-cancellation) at full budget k
      ms_makespan    its post-cancellation makespan
      probe          {lam: probe_makespan} table from the K0 sweep
    Returns None only if SABRE-MS failed at every seed for every lambda.
    """
    # Probe each lambda at the small budget K0 (seeds 0..k0-1).
    probe_seeds = range(k0)
    probe = {}
    for lam in grid:
        mk, _, _ = _best_ms_makespan(qc_phys, graph, coupling, lam, probe_seeds, alim_mult)
        probe[lam] = mk

    # Keep the lambda with the shortest probe makespan. Ties: prefer the smaller
    # lambda (closer to SABRE; deterministic, and lambda=0 wins an exact tie).
    finite = {l: m for l, m in probe.items() if m != float("inf")}
    if not finite:
        return None
    best_lam = min(finite, key=lambda l: (finite[l], l))

    # Final route at the full budget K (seeds 0..k-1) at the chosen lambda.
    ms_mk, ms_qc, ms_raw = _best_ms_makespan(qc_phys, graph, coupling, best_lam,
                                             range(k), alim_mult)
    if ms_qc is None:
        return None
    return {"lam": best_lam, "ms_qc": ms_qc, "ms_makespan": ms_mk,
            "ms_raw": ms_raw, "probe": probe}
