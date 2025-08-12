# pip install huggingface_hub
from huggingface_hub import snapshot_download

# 只下载文件名以 "model-" 开头的文件
snapshot_download(
    repo_id="imone/D4RL",
    repo_type="dataset",  # 指定为数据集类型
    force_download=True,  # 强制下载，覆盖本地同名文件
    # allow_patterns=['walker2d*', 'hopper*', 'halfcheetah*', 'Ant_maze*'],  # 支持通配符 * 和 ?
    allow_patterns=['Ant_maze*'],  # 支持通配符 * 和 ?
    
    # local_dir="/home/robot/.d4rl/datasets",  # 下载到指定目录 ~/.d4rl/datasets
    local_dir="/data/zhouhongtu/D4RL/datasets"
)