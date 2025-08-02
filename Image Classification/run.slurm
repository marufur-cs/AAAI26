#!/bin/bash
#SBATCH --job-name="rahmm224"         # Name that appears in queue
#SBATCH --partition=yangGrp           # Resource group (small/large/gpu)
#SBATCH --account=yangGrp             # Research group
#SBATCH --nodes=1                     # Number of Nodes
#SBATCH --cpus-per-task=1             # Number of CPU Cores
#SBATCH --mem=8G                     # Requested memory (CPU)
#SBATCH --time=00-10:00:00            # Job duration in DD-HH:MM:SS
#SBATCH --mail-user=rahmm224@wfu.edu  # Email address to update on Job status
#SBATCH --mail-type=END,FAIL          # Updates to provide mail-user
#SBATCH --error=/deac/csc/yangGrp/rahmm224/slurm_logs/err_rl-%j.e
#SBATCH --output=/deac/csc/yangGrp/rahmm224/slurm_logs/err_rl-%j.o

# Load your software
# module load python/3.8.13
module load apps/anaconda3/2024.02
conda activate ENV1


echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

dataset="cifar10"
ewc=True
ewc_lambda=0.1
saveRes="mlp_FullMemEwc.csv"
python3 -u /deac/csc/yangGrp/rahmm224/code_imageEx/img2tasksMLP_EWC.py --dataset $dataset \
    --ewc $ewc \
    --ewc_lambda $ewc_lambda \
    --saveRes $saveRes 
    
conda deactivate

