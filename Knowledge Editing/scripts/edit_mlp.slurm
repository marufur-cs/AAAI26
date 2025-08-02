#!/bin/bash
#SBATCH --job-name="rahmm224"         # Name that appears in queue
#SBATCH --partition=gpu               # Resource group (small/large/gpu)
#SBATCH --account=yangGrp             # Research group
#SBATCH --nodes=1                     # Number of Nodes
#SBATCH --cpus-per-task=1             # Number of CPU Cores
#SBATCH --gres=gpu:A100_80:1
#SBATCH --mem=64G                     # Requested memory (CPU)
#SBATCH --time=00-02:00:00            # Job duration in DD-HH:MM:SS
#SBATCH --mail-user=rahmm224@wfu.edu  # Email address to update on Job status
#SBATCH --mail-type=END,FAIL          # Updates to provide mail-user
#SBATCH --error=/deac/csc/yangGrp/rahmm224/slurm_logs/err_rl-%j.e
#SBATCH --output=/deac/csc/yangGrp/rahmm224/slurm_logs/err_rl-%j.o

# Load your software
# module load python/3.8.13
module load apps/anaconda3/2024.02
conda activate ENV1

# cd /scratch/${SLURM_JOB_ID}
# Go to working directory
cd /deac/csc/yangGrp/rahmm224
#conda activate /deac/csc/yangGrp/rahmm224/env1
# model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name="NousResearch/Llama-2-13b-chat-hf"
adapter_type="mlplora"
dataset="zsre"
max_length=32
# sample_per_task=5
# batch=3
save_model="mlp_counterfactTask"
seed=2424
layer_type="query_proj,value_proj"
update_last_layers=2
ewc_lambda=0.1
fisher_mem=4
ewc=True
save_res="llama_mlp.csv"
r=8
samples_batch=("2 1" "3 2" "4 2" "5 3")

for pair in "${samples_batch[@]}"
do
    read s b <<< "$pair"
    sample_per_task=$s
    training_batch=$b
    for ex in {1..5}
    do
        python3 -u code/trainMLP_7b.py \
            --model_name $model_name \
            --adapter_type $adapter_type\
            --layer_type $layer_type\
            --lora_r $r \
            --lora_alpha $((r * 2)) \
            --update_last_layers $update_last_layers\
            --dataset $dataset \
            --max_length $max_length \
            --sample_per_task $sample_per_task\
            --training_batch $training_batch \
            --save_model $save_model \
            --epochs 60 \
            --warmup_steps 1\
            --lr 2e-3\
            --seed $seed\
            --ewc_lambda $ewc_lambda\
            --fisher_mem $fisher_mem\
            --ewc $ewc\
            --save_res $save_res
    done
done
conda deactivate
