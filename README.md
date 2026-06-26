# SABRE-MS: Makespan-Aware Quantum Circuit Routing

Code and released results for the paper
**"Quantum Circuit Routing Optimises the Wrong Metric: Closing the Proxy Gap
Between SWAP Count and Schedule Length"** (CSE3000, TU Delft).

SABRE-MS adds one schedule-aware term to SABRE's routing score so the router
balances the SWAP count against the circuit **makespan** (runtime), instead of
minimising SWAP count alone. A single weight `λ` sets the balance; `λ=0` recovers
production SABRE exactly.

---

## 1. Install

Python 3.11. Exact package versions are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The core makespan/ESP experiments need only `qiskit`, `mqt.bench`, and the
scientific stack. The hardware runs additionally need `qiskit-ibm-runtime` (and an
IBM Quantum account); the RL ablation needs `stable-baselines3` + `sb3-contrib`.

## 2. Reproduce the headline numbers (no recomputation)

Every number in the paper is read from a released result file in `results/`.
To print the headline tables straight from those files:

```bash
python _paper_numbers.py
```

This reproduces: the 20.0% mean makespan reduction (Table 2), the
rescore/algorithm decomposition (Table 3), the channel split, the reachability
result, the ESP ratios (Table 4), the absorption correlation (r=0.93), and the
synthetic-set numbers.

## 3. Re-run an experiment from scratch

Each experiment routes circuits through the full Qiskit pipeline and writes its
result JSON. All seeds are fixed, so runs are deterministic.

```bash
python exp_core_mqt.py        # main makespan comparison (MQT Bench)
python exp_core_esp.py        # ESP reliability evaluation
python exp_real_full.py       # synthetic dataset
python exp_mqt_qubit_sweep.py # scaling sweep (9 -> 49 qubits)
```

## 4. Experiment → result file → paper claim

| Script | Result file | Paper |
|---|---|---|
| `exp_core_mqt.py` | `core_mqt_perrun.json` | Table 2 (20% makespan), Table 3 (decomposition), reachability |
| `exp_core_esp.py` | `core_esp_perrun.json` | Table 4 (ESP, 36/46, median 1.22×) |
| `exp_real_full.py` | `real_full_perrun.json` | synthetic set (13.0%, 24/24 significant) |
| `exp_reranking_mqt.py` | `reranking_mqt.json` | §5.1 proxy-gap figure (Fig. 4) |
| `exp_mqt_channels.py` | `mqt_channels_perrun.json` | channel split (scheduling vs absorption) |
| `exp_swap_absorption.py` | `swap_absorption_perrun.json` | absorption correlation (Fig. 5, r=0.93) |
| `exp_lightsabre_comparison.py` | `lightsabre_comparison_perrun.json` | vs basic/lookahead/decay (672/800) |
| `exp_mqt_qubit_sweep.py` | `mqt_qubit_sweep.json` | scaling figure (to 49 qubits) |
| `exp_mapping_routing_misalignment.py` | `mapping_routing_misalignment.json` | initial-mapping gap (§5.1) |
| `exp_ibm_hardware.py` | `ibm_hw_*.json` | §5.4 hardware (1.7× / 3.6×) |
| `eval_rl_capped35.py` | `rl_capped35.json` | §5.6 RL ablation (8/9) |
| `_paper_numbers.py` | (reads the above) | prints all headline tables |

## 5. Core source files

| File | What it is |
|---|---|
| `sabre_impl.py` | SABRE and **SABRE-MS** routing (the makespan-aware score, Eq. 4) |
| `lambda_select.py` | per-circuit `λ` selection over the grid `{0, .005, .01, .02, .05, .1, .25}` |
| `pipeline.py` | full compile pipeline + ASAP scheduler / makespan |
| `esp.py` | Expected Success Probability model (gate-error × decoherence, Eq. 5) |
| `optimize.py` | gate-cancellation pass |
| `topologies.py` | the six coupling graphs |
| `circuits.py` | synthetic circuit generators (seeded) |

## 6. Hardware runs

The on-device results were run on `ibm_marrakesh` (156-qubit IBM Heron r2). The
IBM job identifiers are recorded in the `ibm_hw_*.json` / `ibm_lambda_sweep_*.json`
result files (the `job_id` field), so the original device jobs can be inspected via
the IBM Quantum platform. Re-running `exp_ibm_hardware.py` requires an IBM Quantum
account and will submit new jobs.

## 7. RL-router ablation

`train_rl.py` trains a qgym MaskablePPO routing agent under a chosen reward
(`--rewarder swapquality` for the baseline, the shaped schedule-aware reward
otherwise). `eval_rl_capped35.py` evaluates the two trained agents and writes
`rl_capped35.json`. Trained model weights are not committed (they regenerate from
training); the result file is released.

## 8. Notes on reproducibility

- **Seeds** are fixed throughout (`np.random.default_rng(seed)` for synthetic
  circuits; `seed=0` for the canonical SabreLayout mapping; explicit trial seeds in
  each routing pool).
- **MQT-Bench** circuits are a deterministic `get_benchmark(...)` library call.
- Cost model: 1 single-qubit gate = 1 cycle, CNOT = 2, SWAP = 6; 1 cycle = 34 ns
  (half the measured median 2-qubit gate time on `ibm_marrakesh`).
- `results/_scratch/` holds superseded/exploratory result files, kept for
  transparency but not used by any paper claim.
