import numpy as np
import minari
from datasets.dataset import Dataset


class D4RLDataset(Dataset):

    def __init__(self,
                 dataset_id: str,
                 download: bool = True
                 ):
        self.dataset = minari.load_dataset(dataset_id, download=download)

        episode_lens, episode_returns = [], []
        observations, actions, rewards, next_observations, terminations, truncations = [], [], [], [], [], []
        for episode in self.dataset:
            observations.append(episode.observations[:-1])
            actions.append(episode.actions)
            rewards.append(episode.rewards)
            next_observations.append(episode.observations[1:])
            terminations.append(episode.terminations)
            truncations.append(episode.truncations)
            episode_returns.append(episode.rewards.sum())
            episode_lens.append(len(episode.rewards))
        
        observations = np.concatenate(observations, axis=0).astype(np.float32)
        actions = np.concatenate(actions, axis=0).astype(np.float32)
        rewards = np.concatenate(rewards, axis=0).astype(np.float32)
        next_observations = np.concatenate(next_observations, axis=0).astype(np.float32)
        terminations = np.concatenate(terminations, axis=0).astype(np.bool_)
        truncations = np.concatenate(truncations, axis=0).astype(np.bool_)
        self.episode_returns = np.array(episode_returns)
        self.episode_lens = np.array(episode_lens)
        

        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=(~terminations).astype(np.float32),
                         dones_float=(terminations | truncations).astype(np.float32),
                         next_observations=next_observations,
                         size=len(rewards),
                         scanning=True)
