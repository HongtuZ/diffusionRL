

#gym
python main.py --dataset_id mujoco/walker2d/expert-v0 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --tag Reproduce --num_seed 1 --gpu '0' --num_qs 10
python main.py --dataset_id walker2d-medium-replay-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id walker2d-medium-expert-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --dataset_id hopper-medium-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1.5 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id hopper-medium-replay-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --rho 1.5 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id hopper-medium-expert-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 0.05 --rho 1.5 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --dataset_id halfcheetah-medium-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1 --q_tar lcb --rho 0 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id halfcheetah-medium-replay-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 1  --q_tar lcb --rho 0 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id halfcheetah-medium-expert-v2 --agent dac --eta 1 --eta_lr 0.001 --bc_threshold 0.1  --q_tar lcb --rho 0 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

#antmaze
python main.py --dataset_id antmaze-umaze-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id antmaze-umaze-diverse-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb --tag Reproduce  --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --dataset_id antmaze-medium-play-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0  --rho 1 --q_tar lcb  --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id antmaze-medium-diverse-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb   --tag Reproduce --num_seed 8 --gpu '0, 1' --num_qs 10

python main.py --dataset_id antmaze-large-play-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0  --rho 1.1 --q_tar lcb  --tag Reproduce   --critic_lr 0.001 --num_seed 8 --gpu '0, 1' --num_qs 10
python main.py --dataset_id antmaze-large-diverse-v0 --agent dac --maxQ --eta 0.1 --eta_lr 0 --rho 1 --q_tar lcb   --tag Reproduce   --critic_lr 0.001 --num_seed 8 --gpu '0, 1' --num_qs 10
