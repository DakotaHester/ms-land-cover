#SBATCH --array=0-164%8
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH -p gpu-a100
#SBATCH -A research-abe
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --job-name hrnet_finetune_tests
#SBATCH --output=./logs/slurm/finetune/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dh2306@msstate.edu



ml cuda
ml python/3.10.8
source $PYTHON_ENV
export CUDA_VISIBLE_DEVICES=0
python $SCRIPT_NAME --num_workers $SLURM_NTASKS
EOL