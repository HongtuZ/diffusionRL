import ray
import numpy as np
from tqdm import tqdm


def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""
    """it is faster to get replace=False"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


def prepare_output_dir(
        folder='results',
        time_stamp=True,
        time_format="%Y%m%d-%H%M%S",
        suffix: str = "") -> str:
    import os
    import datetime
    """Prepare a directory for outputting training results.
    Returns:
        Path of the output directory created by this function (str).
    """

    if time_stamp:
        suffix = str(datetime.datetime.now().strftime(time_format)) + "_" + suffix

    # basedir -> 'results'
    folder = os.path.join(folder or ".", suffix)
    # assert not os.path.exists(out_dir), "found existed experiment folder!"

    os.makedirs(folder, exist_ok=True)

    # Save all the environment variables
    # with open(os.path.join(out_dir, "environ.txt"), "w") as f:
    #     f.write(json.dumps(dict(os.environ)))
    return folder


@ray.remote
class StepMonitor:
    def __init__(self, n_bars: int = 1):
        self.passed_steps = [0] * n_bars

    def update(self, i):
        self.passed_steps[i] += 1

    def get_steps(self):
        return self.passed_steps


class MBars:

    def __init__(self, total: int, title: str, n_bars: int = 1):
        self.max_steps = total
        self.bars = [tqdm(total=total,
                          desc=f'Train {title}[{i}]',
                          smoothing=0.01,
                          colour='GREEN',
                          position=i) for i in range(n_bars)]
        self.process = StepMonitor.remote(n_bars)

    def flush(self):
        import time
        while True:
            for step, bar in zip(ray.get(self.process.get_steps.remote()), self.bars):
                bar.update(step - bar.n)
            time.sleep(0.1)

            if all(bar.n >= self.max_steps for bar in self.bars):
                print('All process finished!')
                return
