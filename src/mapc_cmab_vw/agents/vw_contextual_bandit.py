import numpy as np
import vowpalwabbit


class VwCBAgent:
    """
    Contextual-bandit wrapper with the same public action convention as DQN/RLib:
    actions exposed by this class are 0-based.

    Vowpal Wabbit's contextual-bandit action IDs are 1-based, so the conversion
    is isolated inside this wrapper:
        public/DQN action 0..K-1 <-> VW action 1..K
    """

    def __init__(
        self,
        n_actions: int,
        context_size: int,
        epsilon: float = 0.15,
        seed: int = 42,
    ):
        if n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        if context_size <= 0:
            raise ValueError("context_size must be > 0")

        self.n_actions = n_actions
        self.context_size = context_size
        self.rng = np.random.default_rng(seed)

        self.vw = vowpalwabbit.Workspace(
            f"--cb_explore {n_actions} --epsilon {epsilon}",
            quiet=True,
        )

        # Public convention: 0-based.
        # These values are only used to bootstrap the previous transition.
        self.previous_context = np.zeros(context_size, dtype=np.float32)
        self.previous_action = 0
        self.previous_action_probability = 1.0 / n_actions
        self.has_previous_action = False

    def _validate_context(self, context) -> np.ndarray:
        context = np.asarray(context).reshape(-1)
        if context.size != self.context_size:
            raise ValueError(
                f"Expected context of size {self.context_size}, "
                f"got {context.size}"
            )
        return context

    def _encode_context(self, context) -> str:
        context = self._validate_context(context)
        features = " ".join(
            f"x{i}:{float(value)}"
            for i, value in enumerate(context, start=1)
        )
        return f"| {features}"

    def _encode_learn_example(
        self,
        context,
        action: int,
        reward: float,
        action_probability: float,
    ) -> str:
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"Public action must be in [0, {self.n_actions - 1}], got {action}"
            )

        if action_probability <= 0.0:
            raise ValueError("action_probability must be > 0")

        # DQN/RLib API is 0-based; VW CB action IDs are 1-based.
        vw_action = action + 1

        # VW CB expects cost. The rest of the simulator uses reward.
        cost = -float(reward)

        return (
            f"{vw_action}:{cost:.4f}:{action_probability} "
            f"{self._encode_context(context)}"
        )

    def _learn_previous(self, reward: float) -> None:
        if not self.has_previous_action:
            return

        example = self._encode_learn_example(
            self.previous_context,
            self.previous_action,
            reward,
            self.previous_action_probability,
        )
        self.vw.learn(example)

    def sample(self, context, previous_reward):
        """
        Match the existing hierarchical DQN/RLib calling convention:

            action = agent.sample(
                update_reward_from_previous_action
            )

        Context passed to sample() is the context for the NEW action.
        Reward belongs to the PREVIOUS action and is therefore used before
        predicting the new action.
        """
        self._learn_previous(previous_reward)

        context = self._validate_context(context)
        encoded_context = self._encode_context(context)

        action_probs = np.asarray(self.vw.predict(encoded_context), dtype=np.float64)

        if action_probs.shape != (self.n_actions,):
            raise RuntimeError(
                f"VW returned probabilities with shape {action_probs.shape}; "
                f"expected {(self.n_actions,)}"
            )

        # Position in VW's probability vector is 0-based Python indexing.
        public_action = int(np.argmax(action_probs))

        # Store the public action. Conversion to VW's 1-based ID happens only
        # when the learning example is constructed.
        self.previous_context = context.copy()
        self.previous_action = public_action
        self.previous_action_probability = float(action_probs[public_action])
        self.has_previous_action = True

        return public_action
