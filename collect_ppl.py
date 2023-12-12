import os

import glob
import json


def sort_checkpoints(checkpoints):
    return sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

arch = "ProtLlama2"
total_bs = 2048
grad_acc = 128
seed = 42

output_dir_base = f"/scratch/gpfs/hyen/output"
tag = "initial"
configs = [f"{arch}_51m", f"{arch}_65m", f"{arch}_82m", f"{arch}_97m", f"{arch}_112m", f"{arch}_124m", f"{arch}_146m", f"{arch}_167m"]

final_loss = []
all_results = []

for config in configs:
    for lr in [5e-4, 1e-3, 5e-3]:
        output_dir = os.path.join(output_dir_base, f"{config}_{tag}_lr{lr}_bs{total_bs}_gc{grad_acc}_{seed}")
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        print(f"Found {len(checkpoints)} checkpoints for {config} with lr {lr}")
        checkpoints = sort_checkpoints(checkpoints)

        with open(os.path.join(output_dir, "all_results.json")) as f:
            r = json.load(f)
            final_loss.append({"config": config, "lr": lr, "loss": r["eval_loss"]})
        
        for i, checkpoint in checkpoints:
            with open(os.path.join(checkpoint, "all_results.json")) as f:
                r = json.load(f)
                step = int(checkpoint.split('-')[-1])
                all_results.append({"config": config, "lr": lr, "step": step, "loss": r["eval_loss"]})

print(final_loss)
print(all_results)