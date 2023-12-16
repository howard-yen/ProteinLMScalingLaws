import glob
import math
import torch

from transformers import AutoConfig, AutoModelForCausalLM

arch = "ProtLlama2"
arch = "ProtGPT2"
params = {}
paths = sorted(glob.glob(f"configs/*_*.json"))

dummy = torch.ones(1, 1024, dtype=torch.long)
print("Name,Num parameters (M),Hidden size,Intermediate size,Num attention heads,Num layers")

for p in paths:
    # print(p)
    config = AutoConfig.from_pretrained(p)
    config._flash_attn_2_enabled = False
    m = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    total = sum(p.numel() for p in m.parameters())
    params[p.split("/")[-1].split(".")[0]] = total

    if "ProtGPT2" in p:
        print(f"ProtGPT2,{total / 1e6},{config.n_embd},{config.n_inner},{config.n_head},{config.n_layer}")
    else:
        print(f"ProtLlama2,{total / 1e6},{config.hidden_size},{config.intermediate_size},{config.num_attention_heads},{config.num_hidden_layers}")

params = dict(sorted(params.items(), key=lambda x: x[1]))
print("total params:", params)

