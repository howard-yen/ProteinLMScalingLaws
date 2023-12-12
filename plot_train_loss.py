import os

import glob
import json

import seaborn as sns

arch = "ProtLlama2"
total_bs = 2048
grad_acc = 128
seed = 42

with open(f"{arch}_total_flos.json") as f:
    m = json.load(f)

output_dir_base = f"/scratch/gpfs/hyen/output"
tag = "initial"
configs = [f"{arch}_51m", f"{arch}_65m", f"{arch}_82m", f"{arch}_97m", f"{arch}_112m", f"{arch}_124m", f"{arch}_146m", f"{arch}_167m"]

all_results = []

for config in configs:
    for lr in [5e-4, 1e-3, 5e-3]:
        output_dir = os.path.join(output_dir_base, f"{config}_{tag}_lr{lr}_bs{total_bs}_gc{grad_acc}_{seed}")

        with open(os.path.join(output_dir, "trainer_state.json")) as f:
            r = json.load(f)
            for log in r["log_history"]:
                if "loss" in log:
                    all_results.append({"config": config, "lr": lr, "step": log["step"], "flos": log["step"]*m[config], "loss": log["loss"]})

print(all_results)