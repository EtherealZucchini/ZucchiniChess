import numpy as np
from mcts import MCTSNode as Node

class AlphaZeroConfig(object):
    """
        configuration for AlphaZero from the supplementary materials pseudocode
    """

    def __init__(self):
        ### Self-Play
        self.num_actors = 5000

        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_simulations = 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096

        self.weight_decay = 1e-4
        self.momentum = 0.9
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {
            0: 2e-1,
            100e3: 2e-2,
            300e3: 2e-3,
            500e3: 2e-4
        }


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    child, prior = node.children[a]
    node.children[a] = (child, prior * (1 - frac) + n * frac)
