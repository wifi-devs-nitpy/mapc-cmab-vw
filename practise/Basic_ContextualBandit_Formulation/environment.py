import numpy as np 

class Environment:
    """
        # all the actions considered inside are one based indexing
    """

    def __init__(
        self,
        n_actions: int,
        context_size: int,
        seed=42,
    ):
        self.n_actions = n_actions
        self.context_size = context_size
        self.rng = np.random.default_rng(seed)

    def sample_context(self):
        return self.rng.integers(
            low=-1,
            high=2,
            size=self.context_size,
        )

    def encode_context(self, context: np.array):
        features = " ".join(
            f"x{i}:{value.item()}"

            for i, value in enumerate(context, start=1)
        )

        return f"| {features}"

    def get_reward_array(self, context):
        x1, x2 = context 
        return np.array([
            (-2*x1 -x2),
            (-x1 + 10*x2), 
            (5*x1 - 4*x2), 
            (x1 + 3*x2)            
        ])


    def reward(self, context: tuple, action: int) -> int:
        return self.get_reward_array(context)[action-1] + np.random.normal(0, 0.2)


    def optimal_action(self, context):
        action_0 = np.argmax(self.get_reward_array(context))
        return action_0 + 1

    def encode_learn_example(self, context, action, reward, action_probs):
        cost = -reward 
        action_probability = action_probs[action-1]
        return (
            f"{action}:{cost:.7f}:{action_probability} {self.encode_context(context)}"
            )
