# pip install huggingface_hub
from huggingface_hub import snapshot_download

# 只下载文件名以 "model-" 开头的文件
snapshot_download(
    repo_id="imone/D4RL",
    repo_type="dataset",  # 指定为数据集类型
    allow_patterns=['walker2d*', 'hopper*', 'halfcheetah*', 'Ant_maze*'],  # 支持通配符 * 和 ?
    local_dir="/data/zhouhongtu/D4RL/datasets",  # 下载到指定目录
    local_dir_use_symlinks=False,  # 不创建符号链接（直接下载文件）
    resume_download=True,  # 支持断点续传
)