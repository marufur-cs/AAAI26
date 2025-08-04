from transformers import TrainingArguments, Trainer
import transformers
from data_provider import load_data
from load_model import load_llm
import torch
import torch.nn as nn
from trl import SFTTrainer
import argparse
from transformers import DataCollatorForLanguageModeling, LlamaTokenizer
from evaluateMEditing import eval
import numpy as np
import csv
from editor import *
import time
from tqdm import tqdm

# Create the parser
parser = argparse.ArgumentParser(description="Finetuning llama2 7B")

# Define long-form arguments
parser.add_argument('--model_name', type=str, help="Model name")
parser.add_argument('--continue_training_model', type=str, default="None", help="Previously finetuned model that will be retrained on different task")
parser.add_argument('--adapter_type', type=str, help="mlplora or kanlora")
parser.add_argument('--layer_type', type=str, help="lora layer type")
parser.add_argument('--dataset', type=str, help="metamath, counterfact")
parser.add_argument('--sample_per_task', type=int, help="number of samples per task")
parser.add_argument('--task_num', type=int, help="number of tasks")
parser.add_argument('--max_length', type=int, help="max_length")
parser.add_argument('--training_batch', type=int, help="Batch size")
parser.add_argument('--save_model', type=str, help="path to save finetuned model")
parser.add_argument('--epochs', type=int, help="train epochs")
parser.add_argument('--lr', type=float, help="learning rate")
parser.add_argument('--lora_r', type=int, help="lora rank")
parser.add_argument('--lora_alpha', type=int, help="lora alpha")
parser.add_argument('--warmup_steps', type=int, help="lora alpha")
parser.add_argument('--kan_grid_size', type=int, help="grid size")
parser.add_argument('--kan_update_last_layers', type=int, help="Last layers that will be updated with kan lora adapters")
parser.add_argument('--seed', type=int, help="Random seed")
parser.add_argument('--save_res', type=str, help="Save results to csv file")

#EWC
parser.add_argument('--ewc', type=bool, help="metamath, counterfactual")
parser.add_argument('--ewc_lambda', type=float, help="ewc_lambda")
parser.add_argument('--fisher_mem', type=int, help="fisher_mem")

# Parse the arguments
config = parser.parse_args()
print(vars(config))

print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")


# --- begin editing ---
editor = load_editor(config)
acc = []
batch_history = []
for task_num in range(5):
    config.task_num = task_num
    edit_loader = load_data(config)
    print("Training on task: ", config.task_num)
    for _ in range(config.epochs):
        edit_start = time.time()
        for batch in tqdm(edit_loader):
            tokens = {k: v.to("cuda:0") for k, v in batch.items()}    
            # --- perform edit ---
            editor.edit(tokens, batch_history)

        edit_time = time.time() - edit_start
    
    for batch in tqdm(edit_loader):
        tokens = {k: v.to("cuda:0") for k, v in batch.items()}    
        batch_history.append(tokens) # Append new batch to growing history of edits
    print('Edit time: ', edit_time)
    # Evaluate the model
    acc.append(eval(config, editor.model, task_num))

with open(config.save_res, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(acc)

print('Successful training')





