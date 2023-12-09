import glob
import math
import torch

from transformers import AutoConfig, AutoModelForCausalLM
from fvcore.nn import FlopCountAnalysis

arch = "ProtLlama2"
params = {}
paths = sorted(glob.glob(f"configs/{arch}*"))

dummy = torch.ones(1, 1024)

for p in paths:
    # print(p)
    config = AutoConfig.from_pretrained(p)
    m = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    params[p] = sum(p.numel() if "wte" not in n and "wpe" not in n else 0 for n, p in m.named_parameters())
    if "ProtGPT2" in p:
        print("ProtGPT2", params[p] / 1e6, config.n_embd, config.n_inner, config.n_head, config.n_layer)
    else:
        print("ProtLlama2", params[p] / 1e6, config.hidden_size, config.intermediate_size, config.num_attention_heads, config.num_hidden_layers)

params = dict(sorted(params.items(), key=lambda x: x[1]))
print("total params:", params)

base = f"configs/{arch}_51m.json"
steps = 10000
for k, v in params.items():
    print(k, math.ceil(params[base] / v * steps))
