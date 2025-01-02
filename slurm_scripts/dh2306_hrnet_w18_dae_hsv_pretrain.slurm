#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -p gpu-a100
#SBATCH -A research-abe
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --job-name "dh2306_hrnet_w18_dae_hsv_pretrain"
#SBATCH --output=./log/dh2306_hrnet_w18_dae_hsv_pretrain.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dh2306@msstate.edu

ml cuda
ml python/3.10.8
source .env/bin/activate
python pretrain.py --model hrnet_w18 --pretrain_scheme dae_hsv --pretrain_hdf5_path ./pretrain.hdf5
