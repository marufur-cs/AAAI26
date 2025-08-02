'''
Copied from https://github.com/Thartvigsen/GRACE/blob/main/grace/editors/ft_ewc.py

'''
import torch
from load_model import load_llm
import numpy as np

class load_editor(torch.nn.Module):
    def __init__(self, config):
        """
        This method directly finetunes chosen weights given new inputs
        """
        super(load_editor, self).__init__()
        self.model = load_llm(config)
        self.ewc = config.ewc
        self.ewc_lambda = config.ewc_lambda
        self.fisher_mem = config.fisher_mem
        self.edit_lr = config.lr
        self.trainable_parameter_names = []
        self.trainnable_parameters = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.trainable_parameter_names.append(name)
                self.trainnable_parameters.append(param)
        
        self.opt = torch.optim.AdamW(self.trainnable_parameters,lr=config.lr)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def compute_fisher_matrix(self, batch_history):
        optpar_dict = {}
        fisher_dict = {}
        model_dict = dict(self.model.named_parameters())
        print("Shape of batch history: ", np.array(batch_history).shape)
        for item_num, tokens in enumerate(batch_history[::-1]):
            if item_num < self.fisher_mem:
                outputs = self.model(**tokens)
                logits, loss = outputs.logits, outputs.loss.mean()
                loss.backward()

                for name in self.trainable_parameter_names:
                    if name not in optpar_dict:
                        optpar_dict[name] = model_dict[name].data.clone()
                        fisher_dict[name] = model_dict[name].grad.data.clone().pow(2)
                    else:
                        optpar_dict[name] += model_dict[name].data.clone()
                        fisher_dict[name] += model_dict[name].grad.data.clone().pow(2)
        for name in self.trainable_parameter_names:
            optpar_dict[name] /= self.fisher_mem
            fisher_dict[name] /= self.fisher_mem

        return fisher_dict, optpar_dict

    def edit(self, tokens, batch_history):
        self.model.to("cuda:0")
        self.model.train()
        
        self.losses = []
        if len(batch_history) > 0:
            # Compute Fisher matrix and optimal parameters
            if self.ewc:
                print("Computing Fisher matrix...")
                # Compute Fisher matrix and optimal parameters
                fisher_dict, optpar_dict = self.compute_fisher_matrix(batch_history)
        
        outputs = self.model(**tokens)
        loss = outputs.loss.mean()

        if len(batch_history) > 0 and self.ewc:
            # Add EWC regularization term
            for n, p in zip(self.trainable_parameter_names, self.trainnable_parameters):
                ewc_regularizer = self.ewc_lambda * torch.sum(fisher_dict[n] * (p - optpar_dict[n]) ** 2)
                loss += ewc_regularizer

        # print(f"Loss: {loss.item()}")
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()


        return self.model
