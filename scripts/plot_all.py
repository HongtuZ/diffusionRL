from pathlib import Path
from plot_results import plot_all_runs

if __name__ == "__main__":
    for env_dir in Path("results").glob("*/"):
        if env_dir.is_dir():
            plot_all_runs(str(env_dir), scalar_tag="evaluation/average_means", title="Return", ylabel="Return")
            print(f"Plotted results for {env_dir.name}")
        else:
            print(f"Skipping {env_dir.name}, not a directory.")