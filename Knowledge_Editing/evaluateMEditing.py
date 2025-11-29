from transformers import LlamaTokenizer
import torch
from counterfact import make_datasets
from datasets import load_from_disk
import time

def eval(arg, model, m):
    model_name = arg.model_name
    tokenizer = LlamaTokenizer.from_pretrained(model_name, force_download=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
    model.eval()
    model.to("cuda")
    print(f"Testing after {m+1} trained tasks")
    rs = 0
    for i in range(m+1):
        if arg.dataset=="counterfact":
                dataset_path = "/deac/csc/yangGrp/rahmm224/datasets/counterfact_train_"+str(arg.sample_per_task)+"/set"+str(i)
        elif arg.dataset=="zsre":
                dataset_path = "/deac/csc/yangGrp/rahmm224/datasets/zsre_"+str(arg.sample_per_task)+"/set"+str(i)
        dataset = dataset = load_from_disk(dataset_path)
        
        # Evaluating reliability
        correct_r = 0 
        for j in range(len(dataset['train'])):
                if arg.dataset=="counterfact":
                        prompt = f"Input: {dataset['train'][j]['prompt']}\nResponse: "
                        target_new = dataset['train'][j]['target_new']
                elif arg.dataset=="zsre":
                        prompt = f"Input: {dataset['train'][j]['src']}\nResponse: "
                        target_new = dataset['train'][j]['alt']
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                start = time.time()
                output = model.generate(**inputs,do_sample=False, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id)
                end = time.time()
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                if prompt in generated_text:
                        generated_text = generated_text[len(prompt):].strip()
                correct_r += int(target_new.lower() in generated_text.lower())
                
        r = correct_r/len(dataset['train'])
        rs += r
        print(f"Datasset: {arg.dataset}{arg.sample_per_task}, Task num:{i+1}, Reliability: {r}")
        print(f"Inference time {i+1}: {end - start:.6f} seconds")
    if m > 0:
        return (rs-r)/(m)
    else:
        return 0.0

