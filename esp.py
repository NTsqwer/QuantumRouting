"""Expected/Estimated Success Probability (ESP) for routed qgym circuits.

Two variants, both used in the compilation literature (Nishio et al. 2020,
Murali et al. 2019, Tannu/Qureshi 2019):

1. Gate-error ESP (standard):
   ESP_gate = product(1 - error_per_gate) over all gates in the routed
   circuit, AFTER the optimizer pass. SWAPs are decomposed to 3 CNOTs
   for error accounting, but the optimizer has already absorbed some,
   so we use the post-optimizer 2q gate list.

2. Time-aware ESP (gate errors + decoherence on idle qubits):
   ESP_total = ESP_gate * product over qubits of exp(-idle_time_q / T2)
   where idle_time_q = makespan - busy_time_q (each qubit's idle cycles).

The makespan-aware version is what an experimenter actually cares about:
fewer gates AND shorter makespan both increase ESP.

All durations are in CYCLES (not nanoseconds) -- we use the qgym GD model
(cnot=2, swap=6 cycles). Decoherence is parameterized by T2_cycles, which
converts to a per-cycle idle decay rate of (1 - exp(-1/T2_cycles)).

Default parameters (representative of IBM heron / eagle ~2024-2025):
  cnot_error = 0.007 (0.7% per CNOT)
  swap_error = 1 - (1 - cnot_error)^3   (3 CNOTs per SWAP if NOT absorbed
                                         by optimizer; but we run optimizer
                                         first and just count CNOTs)
  T2_cycles  = 1000 (100 us / 100 ns per cycle = 1000 cycles)

We expose every parameter so the paper can do a sensitivity sweep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optimize import optimize_circuit


CNOT_DUR = 2
SWAP_DUR = 6
ONE_Q_DUR = 1  # single-qubit gate duration in cycles (matches asap_makespan_qc)


@dataclass
class ESPParams:
    """Calibration parameters for ESP computation.

    DEFAULTS = measured ibm_marrakesh (Heron r2) calibration, pulled from the
    backend properties on 2026-06-16:
      median 2q (CZ) error  = 0.286%   -> cnot_error = 0.00286
      median 2q gate length = 68 ns    -> 1 cycle = 34 ns (our CNOT = 2 cycles)
      median T2             = 81 us    -> 81e-6 / 34e-9 = 2382 cycles
      1q error             ~ 0.03%     -> one_q_error = 0.0003
    These replace the earlier guessed defaults (err=0.7%, T2=1000 cycles), which
    overweighted decoherence ~3x and overstated SABRE-MS's ESP gain. The result
    is sensitive to the gate-error / T2 ratio, so report a sensitivity band
    (see device presets below), not just the point value.
    """
    cnot_error: float = 0.00286
    one_q_error: float = 0.0003
    T2_cycles: float = 2382.0   # per-qubit decoherence time, in our cycles
    cnot_duration: int = CNOT_DUR
    swap_duration: int = SWAP_DUR


# Device presets for the sensitivity band. Each is (cnot_error, T2_cycles)
# with T2 already converted to our cycle model (1 cycle = (2q gate ns)/2).
#   marrakesh  : measured Heron r2 (primary).
#   heron_good : an optimistic Heron qubit subset (longer T2).
#   garnet     : IQM Garnet from end-to-end fidelity study (short T2 9.6us,
#                40ns 2q gate -> 1 cycle 20ns -> T2 = 480 cycles), the
#                decoherence-dominated regime where makespan matters most.
DEVICE_PRESETS = {
    "marrakesh":  dict(cnot_error=0.00286, T2_cycles=2382.0),
    "heron_good": dict(cnot_error=0.00100, T2_cycles=5000.0),
    "garnet":     dict(cnot_error=0.00600, T2_cycles=480.0),
}


def _gate_busy_times(gates, n_qubits, params: ESPParams):
    """ASAP schedule the gate list, return per-qubit busy and total cycle counts.

    IMPORTANT: this must match asap_makespan_qc's scheduling exactly --- 1q
    gates take 1 cycle and DO occupy their qubit. Skipping them (as an earlier
    version did) overstates idle time and unfairly penalises the longer-schedule
    method in the decoherence term. Durations: 1q=1, cnot=2 cycles, swap=6.

    Returns:
      busy: array[n_qubits] of cycles each qubit spends executing a gate
      makespan: total schedule length in cycles
    """
    free = np.zeros(n_qubits, dtype=int)
    busy = np.zeros(n_qubits, dtype=int)
    for g in gates:
        if g.name == "cnot":
            dur = params.cnot_duration
        elif g.name == "swap":
            dur = params.swap_duration
        else:
            dur = ONE_Q_DUR  # 1q gate: 1 cycle, occupies the qubit
        if g.q1 == g.q2:
            free[g.q1] += dur
            busy[g.q1] += dur
            continue
        start = max(int(free[g.q1]), int(free[g.q2]))
        free[g.q1] = start + dur
        free[g.q2] = start + dur
        busy[g.q1] += dur
        busy[g.q2] += dur
    makespan = int(free.max()) if len(free) else 0
    return busy, makespan


def esp_gate_only(gates, n_qubits, params: ESPParams = None):
    """Gate-error-only ESP: product of (1 - error) over all gates.

    SWAPs (if any survive optimization) count as 3 CNOTs worth of error.
    1q gates contribute one_q_error each. Uses log-prob for numerical
    stability with deep circuits.
    """
    if params is None:
        params = ESPParams()
    log_p = 0.0
    for g in gates:
        if g.name == "cnot":
            log_p += math.log(1.0 - params.cnot_error)
        elif g.name == "swap":
            # Native SWAP = 3 CNOTs; assume not absorbed since it's still
            # in the circuit. Should rarely happen post-optimization since
            # optimize_circuit decomposes SWAPs.
            log_p += 3.0 * math.log(1.0 - params.cnot_error)
        else:
            # treat anything else as a 1q gate
            log_p += math.log(1.0 - params.one_q_error)
    return math.exp(log_p)


def esp_time_aware(gates, n_qubits, params: ESPParams = None):
    """Full ESP: gate errors + decoherence on idle qubits.

    ESP = product(1 - error_gate_i) * product over qubits of exp(-idle_q / T2)
    where idle_q = makespan - busy_q.

    Returns the ESP value and a breakdown dict.
    """
    if params is None:
        params = ESPParams()
    busy, makespan = _gate_busy_times(gates, n_qubits, params)
    # Gate term: CNOTs, surviving SWAPs (3 CNOTs each), and 1q gates
    log_gate = 0.0
    for g in gates:
        if g.name == "cnot":
            log_gate += math.log(1.0 - params.cnot_error)
        elif g.name == "swap":
            log_gate += 3.0 * math.log(1.0 - params.cnot_error)
        else:
            log_gate += math.log(1.0 - params.one_q_error)
    # Decoherence term
    log_decoh = 0.0
    for q in range(n_qubits):
        idle = max(0, makespan - int(busy[q]))
        log_decoh += -idle / params.T2_cycles
    log_total = log_gate + log_decoh
    return {
        "esp": math.exp(log_total),
        "esp_gate_only": math.exp(log_gate),
        "esp_decoh_only": math.exp(log_decoh),
        "log_esp": log_total,
        "makespan": makespan,
        "total_busy_cycles": int(busy.sum()),
        "total_idle_cycles": int(n_qubits * makespan - busy.sum()),
    }


def esp_full_pipeline(routed_gates, n_qubits, params: ESPParams = None,
                     run_optimizer: bool = True):
    """Convenience: run optimizer, compute time-aware ESP. This is the
    primary metric for evaluating routing+scheduling quality on real
    hardware.

    Note we count CNOTs in the final OPTIMIZED circuit, not in the raw
    routed circuit. SWAPs that get absorbed by the optimizer no longer
    contribute their error penalty.
    """
    if params is None:
        params = ESPParams()
    if run_optimizer:
        final_gates = optimize_circuit(routed_gates, n_qubits)
    else:
        final_gates = routed_gates
    return esp_time_aware(final_gates, n_qubits, params)


if __name__ == "__main__":
    # smoke test
    from qgym.custom_types import Gate
    gates = [Gate("cnot", 0, 1), Gate("cnot", 1, 2), Gate("cnot", 0, 1)]
    p = ESPParams()
    print("3-CNOT circuit ESP (gate-only):", esp_gate_only(gates, 3, p))
    print("3-CNOT circuit ESP (time-aware):", esp_time_aware(gates, 3, p))
