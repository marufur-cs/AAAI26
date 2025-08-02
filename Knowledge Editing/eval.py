from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM

# Create the parser
parser = argparse.ArgumentParser(description="Finetuning llama2 7B")

# Define long-form arguments
parser.add_argument('--model_path', type=str, default="None", help="Path of saved model")
parser.add_argument('--eval_dataset', type=str, help="gsm8k, ")


# Parse the arguments
arg = parser.parse_args()

model_name = "NousResearch/Llama-2-7b-chat-hf"
# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_name, force_download=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

if arg.model_path=="None":
    print("Evaluating base model on dataset-",arg.eval_dataset)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True, 
        device_map="cuda"   
    )

else:
    print("Evaluating pretrained model from-", arg.model_path, "on dataset-",arg.eval_dataset)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained(arg.model_path, quantization_config=quantization_config, device_map="auto", torch_dtype=torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained(arg.model_path, quantization_config=quantization_config, device_map="auto")

    model = torch.load(arg.model_path, weights_only=False)
    model.eval()

task_list = arg.eval_dataset.split(",")

lm = HFLM(pretrained=model, tokenizer=tokenizer)

batch_size = 10
# ,temperature= 0.7,top_p=0.9"
results = evaluator.simple_evaluate(
    model=lm,
    tasks=task_list,
    # num_fewshot = 0,
    # gen_kwargs="do_sample=False",
    batch_size=batch_size,
    confirm_run_unsafe_code=True,
    device="cuda"
    # limit=100
)

# Print and analyze the results

for task, metrics in results["results"].items():
    print(metrics,",")

# commonsense_qa
# gsm8k
# headqa
