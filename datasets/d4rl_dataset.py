import minari.storage
import numpy as np
import minari
from datasets.dataset import Dataset


class D4RLDataset(Dataset):

    def __init__(self,
                 dataset_id: list[str] | str,
                 download: bool = True
                 ):
        if isinstance(dataset_id, str):
            self.dataset = minari.load_dataset(dataset_id, download=download)
        elif isinstance(dataset_id, list):
            if len(dataset_id) == 1:
                self.dataset = minari.load_dataset(dataset_id[0], download=download)
            else:
                sub_ids = [sub_id.split('/')[-1].split('-')[0] for sub_id in dataset_id]
                new_dataset_id = '/'.join(dataset_id[0].split('/')[:-1]) + '/' + '-'.join(sub_ids)
                if minari.storage.get_dataset_path(new_dataset_id).exists():
                    self.dataset = minari.load_dataset(new_dataset_id, download=False)
                else:
                    datasets = []
                    for sub_id in dataset_id:
                        datasets.append(minari.load_dataset(sub_id, download=download))
                    self.dataset = minari.combine_datasets(datasets, new_dataset_id)
        else:
            raise NotImplementedError('Not implement other type of dataset id')

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
