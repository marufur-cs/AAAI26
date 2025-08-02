from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from functools import partial
from kanlora import *
from mlplora import *
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_llm(arg):
    # Load tokenizer and model
    model_name = arg.model_name
    # set_seed(arg.seed)

    pretrained_lora_model_path = arg.continue_training_model

    # model = LlamaForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # Use 8-bit quantization if needed
    #     device_map="auto",   # Automatically map model to GPU
    #     trust_remote_code=True
    # )


    if arg.adapter_type == "mlplora":
        # print("loading mlp lora adapter model")
        # # Load the LoRA-adapted model
        # peft_config = LoraConfig(
        # r = arg.lora_r,
        # lora_alpha = arg.lora_alpha, 
        # target_modules=["q_proj", "v_proj"],
        # lora_dropout=0.1,
        # bias="none", 
        # task_type="CAUSAL_LM"
        # )
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()

        print("loading model with MLP lora adapter")
        # Load the LoRA-adapted model
        model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="cuda"  
            )
        lora_config = {
            'rank': arg.lora_r,
            'alpha': arg.lora_alpha,
            'modules': arg.layer_type.split(",")
        }
        apply_lora = partial(
            MLPLinear,
            rank=lora_config['rank'],
            alpha=lora_config['alpha']
        )
        total_layers = len(model.model.layers)

        for i in range(arg.update_last_layers, 0, -1):
            layer = model.model.layers[total_layers-i]
            if 'query_proj' in lora_config['modules']:
                layer.self_attn.q_proj = apply_lora(layer.self_attn.q_proj)
            if 'value_proj' in lora_config['modules']:
                layer.self_attn.v_proj = apply_lora(layer.self_attn.v_proj)
            if 'down_proj' in lora_config['modules']:
                layer.mlp.down_proj = apply_lora(layer.mlp.down_proj)
            if 'up_proj' in lora_config['modules']:
                layer.mlp.up_proj = apply_lora(layer.mlp.up_proj)
            if 'gate_proj' in lora_config['modules']:
                layer.mlp.gate_proj = apply_lora(layer.mlp.gate_proj)

        # Freeze everything except LoRA layers
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True  # Train only LoRA layers

        print(f"Total parameters: {model.num_parameters()}")
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Trainable percentage: {(trainable_params/model.num_parameters())*100.0}%")
         
    if arg.adapter_type == "kanlora":
        print("loading model with kan lora adapter")
        # Load the LoRA-adapted model
        model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="cuda"  
            )
        lora_config = {
            'rank': arg.lora_r,
            'alpha': arg.lora_alpha,
            'modules': arg.layer_type.split(",")
        }
        apply_lora = partial(
            LoRALinear,
            rank=lora_config['rank'],
            alpha=lora_config['alpha'],
            grid_size = arg.kan_grid_size
        )
        total_layers = len(model.model.layers)

        for i in range(arg.kan_update_last_layers, 0, -1):
            layer = model.model.layers[total_layers-i]
            if 'query_proj' in lora_config['modules']:
                layer.self_attn.q_proj = apply_lora(layer.self_attn.q_proj)
            if 'value_proj' in lora_config['modules']:
                layer.self_attn.v_proj = apply_lora(layer.self_attn.v_proj)
            if 'down_proj' in lora_config['modules']:
                layer.mlp.down_proj = apply_lora(layer.mlp.down_proj)
            if 'up_proj' in lora_config['modules']:
                layer.mlp.up_proj = apply_lora(layer.mlp.up_proj)
            if 'gate_proj' in lora_config['modules']:
                layer.mlp.gate_proj = apply_lora(layer.mlp.gate_proj)
            if 'head' in lora_config['modules']:
                model.lm_head=KANHead(model.lm_head, rank=arg.lora_r, alpha=arg.lora_alpha)

        # Freeze everything except LoRA layers
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True  # Train only LoRA layers

        print(f"Total parameters: {model.num_parameters()}")
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Trainable percentage: {(trainable_params/model.num_parameters())*100.0}%")
                
    if arg.continue_training_model != "None":
        print("Training again")
        model=torch.load(arg.continue_training_model, map_location="cuda",weights_only=False) 


    

    return model