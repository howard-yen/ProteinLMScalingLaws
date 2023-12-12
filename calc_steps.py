import json
import os
import glob
import math

arch = "ProtGPT2"
paths = glob.glob(f"output/{arch}*bs32*")
field = "total_flos"

smallest = 1e30
total_steps = 10000
m = {}
print(paths)
for path in paths:
    with open(os.path.join(path, "trainer_state.json")) as f:
        info = json.load(f)
    value = info["log_history"][-1][field] / 3200
    path = path.split("/")[-1].split("-")[0]
    m[path] = value
    smallest = min(value, smallest)

m = dict(sorted(m.items(), key=lambda x: x[1]))
for k, v in m.items():
    print(k, math.ceil(total_steps * smallest / v))

with open(f"{arch}_{field}.json", "w") as f:
    json.dump(m, f, indent=4)