"""
Lab 1 — A minimal online contextual-bandit experiment with Vowpal Wabbit.

Goal:
    Watch this loop happen explicitly:

        context
          -> VW policy
          -> exploration PMF
          -> sample action
          -> environment reward
          -> (action, cost, propensity) fed back to VW
          -> online update

This lab deliberately uses:
    - 2 binary context features: x0, x1
    - 3 fixed actions: 0, 1, 2
    - a deterministic reward table
    - epsilon-greedy exploration
    - online learning: VW is updated one interaction at a time

Install:
    python -m pip install vowpalwabbit==9.11.2

Vowpal Wabbit 9.11.2 provides Python wheels for CPython 3.14 on Windows x86-64,
Linux x86-64/aarch64 and macOS, according to PyPI.

Important indexing detail:
    VW's non-ADF contextual-bandit label examples use action IDs starting at 1.
    Internally, this script keeps Python actions as 0,1,2 and writes
    action+1 to the VW label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from vowpalwabbit import PredictionType, Workspace
except ImportError as exc:
    raise SystemExit(
        "\nVowpal Wabbit is not installed.\n"
        "Install it with:\n"
        "    python -m pip install vowpalwabbit==9.11.2\n"
        "\nThen run this script again.\n"
    ) from exc


# ---------------------------------------------------------------------------
# 1. A tiny known environment
# ---------------------------------------------------------------------------

N_ACTIONS = 3
EPSILON = 0.20
N_ROUNDS = 2000
PRINT_FIRST_N = 15
PRINT_EVERY = 250
SEED = 7

# Reward[context_index, action]
#
# Contexts:
#   0 -> [0, 0]
#   1 -> [0, 1]
#   2 -> [1, 0]
#   3 -> [1, 1]
#
# Optimal actions:
#   [0,0] -> action 0   (reward 4)
#   [0,1] -> action 1   (reward 5)
#   [1,0] -> action 2   (reward 3)
#   [1,1] -> action 1   (reward 5)
#
# Notice that the best action really depends on context.
REWARD_TABLE = np.array(
    [
        [4.0, 2.0, 3.0],  # [0,0]
        [4.0, 5.0, 3.0],  # [0,1]
        [2.0, 2.0, 3.0],  # [1,0]
        [2.0, 5.0, 3.0],  # [1,1]
    ],
    dtype=float,
)

CONTEXTS = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=int,
)

OPTIMAL_ACTION = np.argmax(REWARD_TABLE, axis=1)
OPTIMAL_REWARD = np.max(REWARD_TABLE, axis=1)


@dataclass
class StepResult:
    round_id: int
    context: tuple[int, int]
    pmf: list[float]
    chosen_action: int
    chosen_probability: float
    reward: float
    cost: float
    optimal_action: int
    optimal_reward: float


def context_to_vw(context: Sequence[int]) -> str:
    """
    Represent a binary context as numerical VW features.

    Example:
        [0, 1] -> "| x0:0 x1:1"

    For the first experiment this is intentionally simple.
    Later we will investigate feature/value representation, namespaces,
    action-dependent features, and interactions.
    """
    x0, x1 = int(context[0]), int(context[1])
    return f"| x0:{x0} x1:{x1}"


def sample_from_pmf(
    pmf: Sequence[float],
    rng: np.random.Generator,
) -> tuple[int, float]:
    """
    Sample an action and return:

        chosen_action_index,
        probability_of_that_action_under_the_behavior_policy

    This is the exact 'propensity' that must be logged/fed back to VW.

    We normalize defensively because floating point arithmetic can produce
    tiny deviations from sum == 1.
    """
    p = np.asarray(pmf, dtype=float)

    if p.ndim != 1 or len(p) != N_ACTIONS:
        raise ValueError(f"Expected {N_ACTIONS} probabilities, got {pmf!r}")

    if np.any(p < -1e-12):
        raise ValueError(f"PMF contains a negative probability: {pmf!r}")

    total = float(p.sum())
    if total <= 0.0:
        raise ValueError(f"PMF sums to {total}, cannot sample from it.")

    p = np.clip(p / total, 0.0, 1.0)

    # Re-normalize after clipping.
    p = p / p.sum()

    action = int(rng.choice(len(p), p=p))
    return action, float(p[action])


def expected_epsilon_greedy_pmf(
    greedy_action: int,
    epsilon: float,
    n_actions: int,
) -> np.ndarray:
    """
    The textbook epsilon-greedy PMF:

        P(greedy) = (1-epsilon) + epsilon/K
        P(other)  = epsilon/K

    We calculate it independently so we can compare the theoretical
    distribution with the distribution VW returns.
    """
    p = np.full(n_actions, epsilon / n_actions, dtype=float)
    p[greedy_action] += 1.0 - epsilon
    return p


def build_vw() -> Workspace:
    """
    Create an online contextual bandit with:

        --cb_explore 3
        --epsilon 0.2

    --cb_explore is the important choice here:
        VW explores online.

    We keep the configuration deliberately boring so that the first lab
    isolates the contextual-bandit mechanics rather than tuning.
    """
    return Workspace(
        arg_list=[
            "--cb_explore",
            str(N_ACTIONS),
            "--epsilon",
            str(EPSILON),
            "--learning_rate",
            "0.5",
            "--power_t",
            "0",
            "--random_seed",
            str(SEED),
            "--quiet",
        ]
    )


def predict_pmf(vw: Workspace, context: Sequence[int]) -> list[float]:
    """
    Ask VW for the full action-probability distribution.

    Modern VW exposes PredictionType.ACTION_PROBS explicitly.
    """
    prediction = vw.predict(
        context_to_vw(context),
        prediction_type=PredictionType.ACTION_PROBS,
    )

    pmf = [float(x) for x in prediction]

    if len(pmf) != N_ACTIONS:
        raise RuntimeError(
            f"VW returned {len(pmf)} action probabilities instead of "
            f"{N_ACTIONS}: {pmf!r}"
        )

    # Helpful sanity check.
    if not np.isclose(sum(pmf), 1.0, atol=1e-6):
        raise RuntimeError(f"VW returned a PMF that does not sum to 1: {pmf!r}")

    return pmf


def main() -> None:
    rng = np.random.default_rng(SEED)
    vw = build_vw()

    total_reward = 0.0
    total_regret = 0.0
    correct_actions = 0

    print("=" * 88)
    print("LAB 1 — ONLINE CONTEXTUAL BANDIT WITH VOWPAL WABBIT")
    print("=" * 88)
    print()
    print("Reward table:")
    print("             A0      A1      A2")
    for i, ctx in enumerate(CONTEXTS):
        print(
            f"  {tuple(ctx)}      "
            f"{REWARD_TABLE[i,0]:5.1f}   "
            f"{REWARD_TABLE[i,1]:5.1f}   "
            f"{REWARD_TABLE[i,2]:5.1f}"
        )

    print()
    print("Known optimal policy:")
    for i, ctx in enumerate(CONTEXTS):
        print(
            f"  context {tuple(ctx)} -> "
            f"A{OPTIMAL_ACTION[i]} "
            f"(expected reward {OPTIMAL_REWARD[i]:.1f})"
        )

    print()
    print(f"epsilon = {EPSILON}")
    print(
        "With 3 actions, when VW has a unique greedy action, "
        f"the theoretical epsilon-greedy PMF is "
        f"{1-EPSILON+EPSILON/N_ACTIONS:.6f} for the greedy action and "
        f"{EPSILON/N_ACTIONS:.6f} for each other action."
    )
    print()
    print(
        "Watch the first rows carefully. The probability column is NOT "
        "an environment output."
    )
    print(
        "It is the probability assigned by VW's behavior policy to the "
        "action that we sampled."
    )
    print()
    print("-" * 88)
    print(
        f"{'t':>4}  {'context':>9}  {'PMF':>26}  {'a':>3}  "
        f"{'p(a|x)':>8}  {'reward':>7}  {'cost':>7}"
    )
    print("-" * 88)

    for t in range(1, N_ROUNDS + 1):
        # In a real system, this context comes from the environment.
        context_index = int(rng.integers(0, len(CONTEXTS)))
        context = CONTEXTS[context_index]

        # ---------------------------------------------------------------
        # Prediction:
        #   context -> VW -> PMF
        # ---------------------------------------------------------------
        pmf = predict_pmf(vw, context)

        # ---------------------------------------------------------------
        # Behavior policy:
        #   PMF -> sampled action + its propensity
        # ---------------------------------------------------------------
        chosen_action, chosen_probability = sample_from_pmf(pmf, rng)

        # ---------------------------------------------------------------
        # Environment:
        #   chosen action -> observed reward
        #
        # We convert reward to cost because VW's CB label is expressed
        # as a cost, where LOWER is better.
        # ---------------------------------------------------------------
        reward = float(REWARD_TABLE[context_index, chosen_action])
        cost = -reward

        optimal_action = int(OPTIMAL_ACTION[context_index])
        optimal_reward = float(OPTIMAL_REWARD[context_index])

        total_reward += reward
        total_regret += optimal_reward - reward
        correct_actions += int(chosen_action == optimal_action)

        # ---------------------------------------------------------------
        # Learning:
        #
        # Python action index:
        #     0,1,2
        #
        # VW --cb K label action ID:
        #     1,2,3
        #
        # So we write chosen_action + 1.
        #
        # The third field is the propensity P(a_t | x_t), NOT the reward.
        # ---------------------------------------------------------------
        vw_action_id = chosen_action + 1

        learn_example = (
            f"{vw_action_id}:{cost}:{chosen_probability} "
            f"{context_to_vw(context)}"
        )

        vw.learn(learn_example)

        if t <= PRINT_FIRST_N:
            print(
                f"{t:4d}  "
                f"{str(tuple(context)):>9}  "
                f"{str([round(x, 4) for x in pmf]):>26}  "
                f"{chosen_action:>3}  "
                f"{chosen_probability:>8.4f}  "
                f"{reward:>7.2f}  "
                f"{cost:>7.2f}"
            )

    print("-" * 88)
    print()

    avg_reward = total_reward / N_ROUNDS
    avg_regret = total_regret / N_ROUNDS
    action_accuracy = correct_actions / N_ROUNDS

    print("FINAL ONLINE METRICS")
    print("--------------------")
    print(f"Average reward:       {avg_reward:.4f}")
    print(f"Average regret:       {avg_regret:.4f}")
    print(f"Greedy-action rate:   {action_accuracy:.4%}")
    print()

    print("POLICY PROBE AFTER LEARNING")
    print("----------------------------")
    print(
        "Here we do NOT sample. We simply inspect VW's PMF for each possible "
        "context and take argmax(PMF) as the currently preferred action."
    )
    print()

    for i, context in enumerate(CONTEXTS):
        pmf = predict_pmf(vw, context)
        greedy = int(np.argmax(pmf))
        print(
            f"context {tuple(context)} | "
            f"PMF={np.round(pmf, 4).tolist()} | "
            f"VW preferred=A{greedy} | "
            f"true best=A{OPTIMAL_ACTION[i]}"
        )

    print()
    print("WHAT YOU SHOULD NOTICE")
    print("----------------------")
    print("1. The context is supplied before every prediction.")
    print("2. VW returns a distribution over actions.")
    print("3. We sample from that distribution ourselves.")
    print("4. The sampled action has a specific propensity p(a|x).")
    print("5. The environment gives only the reward/cost for that chosen action.")
    print("6. We feed VW: action + cost + propensity.")
    print("7. VW updates immediately before the next interaction.")
    print()
    print(
        "Next lab: we will instrument this same experiment to compare "
        "epsilon=0, 0.05, 0.2, 0.5 and measure exploration, regret and "
        "convergence without changing the environment."
    )

    vw.finish()


if __name__ == "__main__":
    main()
