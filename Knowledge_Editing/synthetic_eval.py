from transformers import LlamaTokenizer
import torch
import argparse
from datasets import load_from_disk

# Create the parser
parser = argparse.ArgumentParser(description="Finetuning llama2 7B")
# Define long-form arguments
parser.add_argument('--model_path', type=str, default="None", help="Path of saved model")
parser.add_argument('--eval_dataset', type=str, help="gsm8k, ")
# Parse the arguments
arg = parser.parse_args()

model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name, force_download=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
model = torch.load(arg.model_path, weights_only=False)
model.eval()
model.to("cuda")
tasks=arg.eval_dataset.split(',')
print(f"Evaluating model: {arg.model_path}")
for task in tasks:
    if task == 'stask1':
        dataset = load_from_disk("/deac/csc/yangGrp/rahmm224/synthetic_data/sTask1 Data")
    if task == 'stask2':
        dataset = load_from_disk("/deac/csc/yangGrp/rahmm224/synthetic_data/sTask2 Data")
    correct = 0
    for i in range(len(dataset['test'])):
        prompt = f"### Classify following sample:\n {dataset['test'][i]['sample']}\n\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=1,do_sample=False)
        pred = tokenizer.decode(output[0], skip_special_tokens=True).split('### Response:')[-1].strip()
        if pred==dataset['test'][i]['label']:
            correct+=1
    print(f"Task: {task}, Accuracy: {correct/100.0}")
