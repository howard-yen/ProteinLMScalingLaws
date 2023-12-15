import os

import glob
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model_params import config_to_params

sns.set(font_scale=2.2)

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

all_results = []

for config in configs:
    for lr in ["5e-4", "1e-3", "5e-3"]:
        output_dir = os.path.join(output_dir_base, f"{config}-{tag}-lr{lr}-bs{total_bs}-gc{grad_acc}-{seed}")

        with open(os.path.join(output_dir, "trainer_state.json")) as f:
            r = json.load(f)
            for log in r["log_history"]:
                if "loss" in log:
                    flos = log["step"] * flos_map[config]
                    all_results.append({"config": f"{config.split('_')[0]} {config_to_params[config]:.0f}M","lr": lr, "step": log["step"], "flos": flos, "loss": log["loss"]})

print(all_results)

df = pd.DataFrame(all_results)
df = df[df["flos"] >= flos_min][df["flos"] <= flos_max]
g = sns.relplot(data = df, x="flos", y="loss", hue="config", kind="line", row="lr", aspect=4, facet_kws={"sharey": False, "sharex": True})
g.set_ylabels("train loss")
plt.savefig(f"{arch}_training_curves.png", dpi=500)

