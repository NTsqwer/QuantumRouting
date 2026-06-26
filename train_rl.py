"""V19 training: MaskablePPO + V19 obs (V14-lite + V18 per-edge) + CP reward.

V18 broke linear5/qft (det 43 vs V14-lite 19.66) because it dropped V14-lite's
interaction_gates_ahead + qubit_finish_times features. V19 keeps both feature
sets in the observation. Reward and action masking same as V18.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from qgym.envs.routing import Routing
from qgym.envs.routing.routing_rewarders import SwapQualityRewarder
from qgym.generators.interaction import BasicInteractionGenerator

from action_mask_wrapper import ActionMaskWrapper
from obs_wrapper_v19 import V19ObservationWrapper
from topologies import get as get_topology


GATE_DURATIONS = {"cnot": 2, "swap": 6}


def make_rewarder(name):
    if name == "quality_v9_base":
        return SwapQualityRewarder(illegal_action_penalty=-10.0, penalty_per_swap=-2.0,
                                    reward_per_surpass=2.0, good_swap_reward=2.0)
    if name == "aware_critical_path_asap":
        from v12_rewarder import PotentialShapedRewarder
        return PotentialShapedRewarder(
            gate_durations=GATE_DURATIONS,
            illegal_action_penalty=-10.0, penalty_per_swap=-2.0,
            reward_per_surpass=2.0, good_swap_reward=0.0,
            potential_lambda=1.0, gamma=1.0,
            potential_kind="critical_path",
            final_makespan_weight=0.3,
            optimize_terminal=True, realistic_hardware=None,
        )
    raise ValueError(name)


def make_env(topology, rewarder_name, max_circuit_length=10, seed=0,
             max_episode_steps=200, mix_weights=None, sabre_layout=True):
    graph, _ = get_topology(topology)
    from mixed_generator import MixedInteractionGenerator
    gen = MixedInteractionGenerator(
        max_length=max_circuit_length, seed=seed,
        weights=mix_weights or {"random": 0.25, "qft": 0.25,
                                 "parallel": 0.25, "trotter": 0.25},
    )
    if sabre_layout:
        from sabre_layout_generator import SabreLayoutGeneratorWrapper
        gen = SabreLayoutGeneratorWrapper(gen, seed=seed)
    env = Routing(
        connection_graph=graph,
        interaction_generator=gen,
        max_observation_reach=max_circuit_length,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        rewarder=make_rewarder(rewarder_name),
    )
    env = V19ObservationWrapper(env)
    env = ActionMaskWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return Monitor(env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", required=True)
    p.add_argument("--rewarder", required=True,
                   choices=["quality_v9_base", "aware_critical_path_asap"])
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--init-from", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net-arch", default="256,256")
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-circuit-length", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--mix-weights", default=None)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mix_weights = None
    if args.mix_weights:
        mix_weights = {}
        for kv in args.mix_weights.split(","):
            k, v = kv.split(":")
            mix_weights[k.strip()] = float(v.strip())

    env = make_env(
        args.topology, args.rewarder,
        max_circuit_length=args.max_circuit_length, seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        mix_weights=mix_weights,
    )
    net_arch = [int(s) for s in args.net_arch.split(",")]

    if args.init_from:
        print(f"Warm-starting from {args.init_from}")
        model = MaskablePPO.load(
            args.init_from, env=env,
            learning_rate=args.learning_rate, n_steps=args.n_steps,
            batch_size=args.batch_size, ent_coef=args.ent_coef, gamma=args.gamma,
        )
    else:
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy, env, verbose=1, seed=args.seed,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, ent_coef=args.ent_coef,
            gamma=args.gamma, policy_kwargs={"net_arch": net_arch},
        )

    print(f"Training {args.rewarder} on {args.topology} for {args.steps} steps "
          f"(MaskablePPO + V19 obs)")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
