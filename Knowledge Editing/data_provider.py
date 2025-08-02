from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from counterfact import make_datasets
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling

def format_gsm8k(example):
    return {
        "text": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n {example['question']}\n\n### Response: {example['answer']}"
    }
def format_mbpp(example):
    return {
        "text": f"### Instruction:\n {example['prompt']}\n\n### Test cases: {example['test_list']}\n\n### Response: {example['code']}"
    }

def format_commonsense_qa(example):
    choices = "\n".join([f"{label}: {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])])
    prompt = (
        f"Question: {example['question']}\n"
        f"Choices:\n{choices}\n"
        f"Answer: {example['answerKey']}"
    )
    return {"text": prompt}

def format_head_qa(example):
    choices = "\n".join([f"{option['aid']}: {option['atext']}" for option in example['answers']])
    prompt = (
        f"Question Category: {example['category']}\n"
        f"Question: {example['qtext']}\n"
        f"Choices:\n{choices}\n"
        f"Answer: {example['ra']}"
    )
    return {"text": prompt}

def format_synthetic_example(example):
    return {
        "text": f"### Classify following sample:\n {example['sample']}\n\n### Response: {example['label']} </s>"
    }
def format_c(example):
    return {
        "text": f"Input: {example['prompt']}\nResponse: {example['target_new']}</s>"
    }
def format_z(example):
    return {
        "text": f"Input: {example['src']}\nResponse: {example['alt']}</s></s>"
    }
def format_c_pretrain(example):
    return {
        "text": f"Input: {example['prompt']}\nResponse: {example['ground_truth']}</s>"
    }

def tokenize_func(examples, tokenizer, arg):
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=arg.max_length)
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_with_loss_mask(example, tokenizer, arg):
    split_prompt = example["text"].split("Response:", maxsplit=1)
    # print(example["text"])
    # print(split_prompt)
    input_part = split_prompt[0] + "Response: "  # keep "Response:" attached
    response_part = split_prompt[1].strip() 

    full_text = example["text"]
    
    encoded = tokenizer(full_text, padding="max_length", truncation=True, max_length=arg.max_length)
    prompt_encoded = tokenizer(input_part, add_special_tokens=False)["input_ids"]
    
    labels = encoded["input_ids"].copy()
    prompt_length = len(prompt_encoded)
    labels[:prompt_length] = [-100] * prompt_length
    # attention = sum(encoded["attention_mask"])
    # labels[attention:] = [-100] * (len(labels) - attention)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }

def tokenize_synthetic_function(examples, tokenizer):
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=30)
    result["labels"] = result["input_ids"].copy()
    return result

def load_data(config):
    tokenizer = LlamaTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    if config.dataset == "gsm8k":
        print('Loading gsm8k dataset')
        ## Dataset preparation
        dataset = load_dataset("gsm8k",'main')

        formatted_data = dataset.map(format_gsm8k, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_func(x, tokenizer), batched=True)

        return tokenized_dataset
    
    if config.dataset == "mbpp":
        print('Loading mbpp dataset')
        ## Dataset preparation
        dataset = load_dataset("mbpp",'sanitized')

        formatted_data = dataset.map(format_mbpp, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_func(x, tokenizer), batched=True)

        return tokenized_dataset
    
    if config.dataset == "commonsense_qa":
        print('Loading commonsense_qa dataset')
        ## Dataset preparation
        dataset = load_dataset("commonsense_qa")

        formatted_data = dataset.map(format_commonsense_qa, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_func(x, tokenizer), batched=True)

        return tokenized_dataset
    
    if config.dataset == "head_qa":
        print('Loading head_qa dataset')
        ## Dataset preparation
        dataset = load_dataset('head_qa', 'en', trust_remote_code=True)

        formatted_data = dataset.map(format_head_qa, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_func(x, tokenizer), batched=True)

        return tokenized_dataset
    
    if config.dataset == "stask1":
        print('Loading synthetic sTask1 dataset')
        ## Dataset preparation
        dataset = load_from_disk("/deac/csc/yangGrp/rahmm224/synthetic_data/sTask1 Data")

        formatted_data = dataset.map(format_synthetic_example, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_synthetic_function(x, tokenizer), batched=True)

        return tokenized_dataset
    
    if config.dataset == "stask2":
        print('Loading synthetic sTask2 dataset')
        ## Dataset preparation
        dataset = load_from_disk("/deac/csc/yangGrp/rahmm224/synthetic_data/sTask2 Data")

        formatted_data = dataset.map(format_synthetic_example, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_synthetic_function(x, tokenizer), batched=True)

        return tokenized_dataset

    
    if config.dataset == "counterfact":

        print(f'Loading counterfact dataset with {config.sample_per_task} samples per task, Task {config.task_num}')
        ## Dataset preparation
        dataset_path = "/deac/csc/yangGrp/rahmm224/datasets/counterfact_train_"+str(config.sample_per_task)+"/set" + str(config.task_num)
        dataset = load_from_disk(dataset_path)

        formatted_data = dataset.map(format_c, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_with_loss_mask(x, tokenizer, config), batched=False, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        dataLoader = DataLoader(tokenized_dataset["train"], batch_size=config.training_batch, shuffle=True, collate_fn=data_collator)
        return dataLoader
    
    if config.dataset == "zsre":

        print(f'Loading ZSRE dataset with {config.sample_per_task} samples per task')
        ## Dataset preparation
        dataset_path = "/deac/csc/yangGrp/rahmm224/datasets/zsre_"+str(config.sample_per_task)+"/set" + str(config.task_num)
        dataset = load_from_disk(dataset_path)

        formatted_data = dataset.map(format_z, remove_columns=dataset["train"].column_names)
        tokenized_dataset = formatted_data.map(lambda x: tokenize_with_loss_mask(x, tokenizer, config), batched=False, remove_columns=["text"])
        # print(tokenized_dataset)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        dataLoader = DataLoader(tokenized_dataset["train"], batch_size=config.training_batch, shuffle=True, collate_fn=data_collator)
        return dataLoader

