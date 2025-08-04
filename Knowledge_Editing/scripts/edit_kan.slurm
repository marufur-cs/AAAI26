#!/bin/bash
#SBATCH --job-name="rahmm224"         # Name that appears in queue
#SBATCH --partition=gpu           # Resource group (small/large/gpu)
#SBATCH --account=yangGrp             # Research group
#SBATCH --nodes=1                     # Number of Nodes
#SBATCH --cpus-per-task=1             # Number of CPU Cores
#SBATCH --gres=gpu:A100_80:1                  # A100_80/A100_40/V100_32, gres=gpu:A100_80:1
#SBATCH --mem=8G                     # Requested memory (CPU)
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
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#self_attention (query_proj,value_proj), mlp (down_proj,up_proj,gate_proj), head
seed=2424
# model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name="NousResearch/Llama-2-7b-chat-hf"
model_name="NousResearch/Llama-2-13b-chat-hf"
dataset="zsre"
# sample_per_task=5
# training_batch=3
kan_grid_size=5

max_length=32
adapter_type="kanlora"
layer_type="query_proj,value_proj"
lora_r=8
lora_alpha=16
kan_update_last_layers=2
epochs=60
warmup_steps=1 
lr=2e-3
save_model="kan_Task"
ewc=True
ewc_lambda=0.1
fisher_mem=4
save_res="llama_kan.csv"
samples_batch=("2 1" "3 2" "4 2" "5 3")

for pair in "${samples_batch[@]}"
do
    read s b <<< "$pair"
    sample_per_task=$s
    training_batch=$b
    for ex in {1..5}
    do
        python3 -u /deac/csc/yangGrp/rahmm224/code/trainKAN_EWC_llama7b.py \
            --model_name $model_name \
            --adapter_type $adapter_type\
            --layer_type $layer_type \
            --lora_r $lora_r \
            --lora_alpha $lora_alpha \
            --dataset $dataset \
            --sample_per_task $sample_per_task \
            --max_length $max_length \
            --training_batch $training_batch \
            --save_model $save_model \
            --epochs $epochs \
            --warmup_steps $warmup_steps\
            --lr $lr \
            --kan_grid_size $kan_grid_size \
            --kan_update_last_layers $kan_update_last_layers\
            --ewc $ewc\
            --ewc_lambda $ewc_lambda \
            --fisher_mem $fisher_mem \
            --save_res $save_res \
            --seed $seed
    done
done
conda deactivate

