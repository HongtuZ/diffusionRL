#!/bin/bash

# 检查参数数量
if [ $# -ne 3 ]; then
    echo "Usage: $0 <env> <gpu> <agent>"
    echo "Example: $0 hopper 2 dac"
    exit 1
fi

env=$1
agent=$3
gpu=$2
num_seed=4
clean_q_std_k=2.0

case $env in
    "walker")
        python main.py --env walker2d-medium-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb  --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env walker2d-expert-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 0.1 --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env walker2d-medium-replay-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env walker2d-medium-expert-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        ;;
    "hopper")
        python main.py --env hopper-medium-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1.5 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env hopper-expert-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 0.05 --rho 1.5 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env hopper-medium-replay-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1.5 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env hopper-medium-expert-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 0.05 --rho 1.5 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        ;;
    "halfcheetah")
        echo "Running experiments for HalfCheetah"
        python main.py --env halfcheetah-medium-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1 --q_tar lcb --rho 0 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env halfcheetah-expert-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 0.1 --q_tar lcb --rho 0 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env halfcheetah-medium-replay-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 1  --q_tar lcb --rho 0 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env halfcheetah-medium-expert-v2 --agent $agent --eta 1 --eta_lr 0.001 --bc_threshold 0.1  --q_tar lcb --rho 0 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        ;;
    "antmaze")
        python main.py --env antmaze-umaze-v0 --agent $agent --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env antmaze-umaze-diverse-v0 --agent $agent --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env antmaze-medium-play-v0 --agent $agent --maxQ --eta 0.1 --eta_lr 0  --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env antmaze-medium-diverse-v0 --agent $agent --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env antmaze-large-play-v0 --agent $agent --maxQ --eta 0.1 --eta_lr 0  --rho 1.1 --q_tar lcb --critic_lr 0.001 --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        python main.py --env antmaze-large-diverse-v0 --agent $agent --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --critic_lr 0.001 --num_seed $num_seed --gpu "$gpu" --num_qs 10 --clean_q_std_k $clean_q_std_k
        ;;
    *)
        echo "Unknown environment: $env should be one of {walker, hopper, halfcheetah, antmaze}"
        exit 1
        ;;
esac