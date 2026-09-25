import numpy as np 
import jax 
import jax.numpy as jnp 
import vowpalwabbit 


class VwCBAgent:

    def __init__(
        self,
        n_actions: int,
        context_size: int,
        epsilon: float = 0.15,
        seed: int = 42,
    ):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        self.vw = vowpalwabbit.Workspace(
            f"--cb_explore {n_actions} --epsilon {epsilon}",
            quiet=True,
        )

        self.previous_context = np.zeros(context_size)
        self.previous_action = 1
        self.previous_action_probability = 1/self.n_actions

    def _encode_context(self, context):
        features = " ".join(
            f"x{i}:{value.item()}" 

            for i, value in enumerate(context, start=1)
        )

        return f"| {features}"

    def _encode_learn_example(self, context, action, reward, action_probability):
            cost = -reward 
            return (
                f"{action}:{cost:.4f}:{action_probability} {self._encode_context(context)}"
                )

    def _learn(self, reward):
        context = self.previous_context
        action = self.previous_action
        probability = self.previous_action_probability

        example = self._encode_learn_example(context, action, reward, probability)
        self.vw.learn(example)

    def sample(self, context, previous_reward):
        
        self._learn(previous_reward)

        encoded_context = self._encode_context(context)

        action_probs = self.vw.predict(encoded_context)

        action_index = np.argmax(action_probs)
        action = action_index + 1 

        self.previous_context = context.copy()
        self.previous_action = action
        self.previous_action_probability = action_probs[action_index]

        return action