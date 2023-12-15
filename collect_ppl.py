import os

import glob
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model_params import config_to_params

sns.set(font_scale=2.2)

def sort_checkpoints(checkpoints):
    return sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))

arch = "ProtGPT2"
total_bs = 2048
grad_acc = 128
seed = 42

flos_min = 0
flos_max = 8e14
with open(f"{arch}_total_flos.json") as f:
    flos_map = json.load(f)

output_dir_base = f"output"
tag = "v2"
configs = [f"{arch}_51m", f"{arch}_65m", f"{arch}_82m", f"{arch}_97m", f"{arch}_112m", f"{arch}_124m", f"{arch}_146m", f"{arch}_167m"]

final_loss = []
all_results = []
best_final_loss = {}

for config in configs:
    for lr in ["5e-4", "1e-3", "5e-3"]:
        output_dir = os.path.join(output_dir_base, f"{config}-{tag}-lr{lr}-bs{total_bs}-gc{grad_acc}-{seed}")
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        print(f"Found {len(checkpoints)} checkpoints for {config} with lr {lr}")
        checkpoints = sort_checkpoints(checkpoints)

        with open(os.path.join(output_dir, "all_results.json")) as f:
            r = json.load(f)
            final_loss.append({"config": config, "lr": lr, "loss": r["eval_loss"], "output_dir": os.path.abspath(output_dir)})
            if config not in best_final_loss or best_final_loss[config]["loss"] > r["eval_loss"]:
                best_final_loss[config] = final_loss[-1]

        for i, checkpoint in enumerate(checkpoints):
            with open(os.path.join(checkpoint, "all_results.json")) as f:
                r = json.load(f)
                step = int(checkpoint.split('-')[-1])
                flos = step * flos_map[config]
                all_results.append({"config": f"{config.split('_')[0]} {config_to_params[config]:.0f}M", "lr": lr, "step": step, "flos": flos, "loss": r["eval_loss"]})

yaml_template = """
representation_name: {config}
benchmark: all
representation_file_human: {output_dir}/encoded_protein_sequence_records_df.csv
representation_file_affinity: {output_dir}/encoded_SKEMPI_seq.csv
similarity_tasks: ["Sparse","200","500"]
function_prediction_aspect: All_Aspects
function_prediction_dataset: All_Data_Sets
family_prediction_dataset: ["nc","uc50","uc30","mm15"]
detailed_output: False
"""

for config, v in best_final_loss.items():
    with open(f"PROBE/bin/{config}.yaml", "w") as f:
        f.write(yaml_template.format(**v))

print("making figure...")
df = pd.DataFrame(all_results)
df = df[df["flos"] >= flos_min][df["flos"] <= flos_max]
g = sns.relplot(data = df, x="flos", y="loss", hue="config", kind="line", row="lr", aspect=4, facet_kws={"sharey": False, "sharex": True})
g.set_ylabels("eval loss")
plt.savefig(f"{arch}_eval_curves.png", dpi=500)
print("saved figure")
