import argparse

import jax
import numpy as np
from tqdm import tqdm

from mapc_cmab_vw.agents.mapc_vw_cb_agent_factory import MapcVwCBAgentFactory
from mapc_cmab_vw.envs.scenario_impl import (
    residential_scenario,
    small_office_scenario,
    small_office_scenario_rotated,
)
from mapc_cmab_vw.plots.throughput_analysis.throughput_ci import (
    analyze_and_plot_throughputs,
)


d_ap = 10
n_steps = 10_000


class MixScen:
    """
    Non-stationary scenario:
      first half  -> d_sta_1
      second half -> d_sta_2

    The bandit agent is not reset when the scenario changes, so the experiment
    measures how well it adapts to a changed environment.
    """

    def __init__(
        self,
        scenario_factory,
        d_sta_1: int,
        d_sta_2: int,
        d_ap=d_ap,
        max_steps: int = n_steps,
    ):
        self.scen1 = scenario_factory(d_ap=d_ap, d_sta=d_sta_1)
        self.scen2 = scenario_factory(d_ap=d_ap, d_sta=d_sta_2)

        self.step = 0
        self.switch_steps = max_steps // 2

        self.data_rate_fn1 = jax.jit(self.scen1.data_rate_fn)
        self.data_rate_fn2 = jax.jit(self.scen2.data_rate_fn)

        self.data_rate_fn = self.data_rate_fn1
        self.associations = self.scen1.associations

        self.str_repr = (
            f"mix_scen_ap_{d_ap}_"
            f"dsta_{d_sta_1}_{d_sta_2}_"
            f"s{max_steps}"
        )

    def __call__(self, key, link_ap_sta):
        self.step += 1

        if self.step == self.switch_steps:
            self.data_rate_fn = (
                self.data_rate_fn2
                if self.data_rate_fn is self.data_rate_fn1
                else self.data_rate_fn1
            )

        return self.data_rate_fn(key, link_ap_sta=link_ap_sta)

    def reset(self):
        self.data_rate_fn = self.data_rate_fn1
        self.step = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sequential hierarchical VW contextual-bandit MAPC experiments."
    )

    parser.add_argument("--d-ap", type=float, default=10.0)
    parser.add_argument("--d-sta", type=float, default=2.0)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--n-links", type=int, default=3)
    parser.add_argument("--n-tx-power-levels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-duration", type=float, default=0.005)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--show", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# VW contextual-bandit parameters.
#
# VwCBAgent exposes the same public 0-based action API as the DQN agents.
# The wrapper converts public action [0, K-1] to VW action [1, K] internally.
#
# Keep one parameter dictionary per hierarchy level so they can be tuned
# independently later.
# ---------------------------------------------------------------------------

agent_params_lvl1 = {
    "epsilon": 0.15,
}

agent_params_lvl2 = {
    "epsilon": 0.15,
}

agent_params_lvl3 = {
    "epsilon": 0.15,
}

agent_params_lvl4 = {
    "epsilon": 0.15,
}


def create_agent_factory(scenario, args, seed):
    return MapcVwCBAgentFactory(
        associations=scenario.associations,
        agent_params_lvl1=agent_params_lvl1,
        agent_params_lvl2=agent_params_lvl2,
        agent_params_lvl3=agent_params_lvl3,
        agent_params_lvl4=agent_params_lvl4,
        n_tx_power_levels=args.n_tx_power_levels,
        n_links=args.n_links,
        seed=seed,
    )


def run_single_experiment(
    agent_factory,
    scenario,
    run_number,
    n_steps,
    key,
):
    """
    Run one independent VW contextual-bandit experiment.

    Reward is one step delayed:
        agent.sample(previous_throughput)
            -> learns the previous action
            -> samples the next action

        environment(next_action)
            -> produces current throughput

        current throughput becomes previous_throughput for the next step.
    """
    agent = agent_factory.create_hierarchical_vw_cb_agent(logger=None)

    throughputs = np.zeros(n_steps, dtype=np.float32)
    previous_throughput = 0.0

    scenario.reset()

    # Same indexing convention as the existing DQN evaluator.
    for step in tqdm(
        range(1, n_steps),
        desc=f"run_number: {run_number}",
        leave=True,
    ):
        key, step_key = jax.random.split(key)

        # Same public API as HierarchicalMapcDQNAgent.
        tx_config = agent.sample(reward=previous_throughput)

        data_rate = scenario(step_key, tx_config)
        data_rate = float(data_rate)

        throughputs[step] = data_rate
        previous_throughput = data_rate

    return throughputs


def run_experiments(scenario, args):
    throughputs = np.zeros(
        (args.n_runs, args.n_steps),
        dtype=np.float32,
    )

    run_keys = jax.random.split(
        jax.random.PRNGKey(args.seed),
        args.n_runs,
    )

    for run_number in tqdm(
        range(args.n_runs),
        desc="Running VW contextual-bandit experiments",
    ):
        # Fresh factory/agent population for every independent run.
        run_seed = args.seed + run_number

        agent_factory = create_agent_factory(
            scenario=scenario,
            args=args,
            seed=run_seed,
        )

        throughputs[run_number] = run_single_experiment(
            agent_factory=agent_factory,
            scenario=scenario,
            run_number=run_number,
            n_steps=args.n_steps,
            key=run_keys[run_number],
        )

    return throughputs


def main():
    args = parse_args()

    # JAX persistent compilation cache.
    jax.config.update(
        "jax_compilation_cache_dir",
        "./jax_cache",
    )
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "all",
    )

    # ---------------------------------------------------------------
    # Scenario selection
    # ---------------------------------------------------------------

    # scenario = small_office_scenario(
    #     d_ap=args.d_ap,
    #     d_sta=args.d_sta,
    #     n_tx_power_levels=args.n_tx_power_levels,
    # )

    # scenario = residential_scenario(
    #     x_apartments=5,
    #     y_apartments=3,
    #     n_sta_per_ap=4,
    #     size=10,
    #     seed=args.seed,
    # )

    # Same non-stationary scenario as the DQN experiment:
    # first half: d_sta = 2
    # second half: d_sta = 4
    scenario = MixScen(
        small_office_scenario,
        2,
        4,
        args.d_ap,
        max_steps=args.n_steps,
    )

    filename = args.filename
    if filename is None:
        filename = (
            f"vw_cb_"
            f"{scenario.str_repr}_"
            f"runs_{args.n_runs}_"
            f"steps_{args.n_steps}"
        )

    throughputs = run_experiments(
        scenario=scenario,
        args=args,
    )

    analyze_and_plot_throughputs(
        throughputs=throughputs,
        window_size=args.window_size,
        step_duration=args.step_duration,
        confidence=args.confidence,
        output_dir=args.output_dir,
        filename=filename,
        show=args.show,
    )


if __name__ == "__main__":
    main()
