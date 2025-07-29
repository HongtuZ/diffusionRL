from typing import Tuple

import gymnasium as gym

from datasets.d4rl_dataset import D4RLDataset
from datasets.dataset import Dataset
from typing import Callable
import wrappers
import numpy as np


def make_env_and_dataset(seed: int, dataset_name: list | str,
                         video_save_folder: str = None, reward_tune: str = 'no',
                         episode_return: bool = False, scanning: bool = True) -> Tuple[gym.Env, Dataset, Callable]:
    print('Loading dataset:', dataset_name)
    dataset = D4RLDataset(dataset_name, download=True)

    eval_env = dataset.dataset.recover_environment(eval_env=True)
    eval_env = wrappers.EpisodeMonitor(eval_env)  # record info['episode']['return', 'length', 'duration']
    eval_env = wrappers.SinglePrecision(eval_env)  # -> np.float32

    if video_save_folder is not None:
        eval_env = gym.wrappers.RecordVideo(eval_env, video_save_folder)

    # set seeds
    eval_env.action_space.seed(seed)
    eval_env.observation_space.seed(seed)

    # reward normalization
    if reward_tune == 'antmaze100':
        def tune_fn(r):
            return r * 100
    elif reward_tune == 'iql_locomotion':
        # iql_normalize: normalized reward <- reward /(max_return-min_return)* 1000.0
        # seed https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py
        def tune_fn(r):
            return 1000.0 * r / np.ptp(dataset.episode_returns)
    elif reward_tune == 'traj_normalize':
        def tune_fn(r):
            return dataset.episode_lens.mean() * r / np.ptp(dataset.episode_returns)
    elif reward_tune == 'reward_normalize':
        r_mean, r_std = dataset.rewards.mean(), dataset.rewards.std()
        def tune_fn(r):
            return (r - r_mean) / r_std
    elif reward_tune == 'iql_antmaze':
        def tune_fn(r):
            return r - 1.0
    elif reward_tune == 'cql_antmaze':
        def tune_fn(r):
            return (r - 0.5) * 4.0
    elif reward_tune == 'antmaze':
        def tune_fn(r):
            return (r - 0.25) * 2.0
    else:
        tune_fn = None

    if tune_fn is not None:
        dataset.rewards = tune_fn(dataset.rewards)

    return eval_env, dataset, tune_fn
