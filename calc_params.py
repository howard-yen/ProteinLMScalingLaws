import glob
import math
from transformers import AutoConfig, AutoModelForCausalLM

params = {}
paths = glob.glob("configs/*")
for p in paths:
    print(p)
    config = AutoConfig.from_pretrained(p)
    m = AutoModelForCausalLM.from_config(config)
    params[p] = sum(p.numel() if "wte" not in n and "wpe" not in n else 0 for n, p in m.named_parameters())

params = dict(sorted(params.items(), key=lambda x: x[1]))
print("total params:", params)

base = "configs/ProtGPT2_51m.json"
steps = 10000
for k, v in params.items():
    print(k, math.ceil(params[base] / v * steps))
