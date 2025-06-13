import numpy as np
from utils import sample_n_k
from networks.types import Batch

class Dataset(object):
    """
    mask = 1 - terminal, which is used for Q <- r + mask*(discount*Q_next)
    """

    def __init__(self,
                 observations: np.ndarray,
                 actions: np.ndarray,
                 rewards: np.ndarray,
                 masks: np.ndarray,
                 dones_float: np.ndarray,
                 next_observations: np.ndarray,
                 size: int,
                 scanning: bool = False,  # scanning=True for offlineRL, may not be suitable for online replay buffer
                 ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

        self.scanning = scanning
        if self.scanning:
            self.scanning_indices = np.arange(self.size)
            np.random.shuffle(self.scanning_indices)
            self.batch_idx = 0

    def sample(self, batch_size: int) -> Batch:

        if self.scanning:
            indices = self.scanning_indices[self.batch_idx:self.batch_idx + batch_size]
            self.batch_idx += batch_size
            if self.batch_idx >= self.size:
                np.random.shuffle(self.scanning_indices)
                self.batch_idx = 0
        else:
            indices = sample_n_k(self.size, batch_size)

        return Batch(observations=self.observations[indices],
                     actions=self.actions[indices],
                     rewards=self.rewards[indices],
                     masks=self.masks[indices],
                     next_observations=self.next_observations[indices],)
                     # mc_return=self.mc_return[indices])
