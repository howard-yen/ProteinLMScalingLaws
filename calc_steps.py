import json
import os
import glob
import math

arch = "ProtGPT2"
paths = glob.glob(f"output/{arch}*bs32*")
field = "train_runtime"

smallest = 1e30
total_steps = 10000
map = {}
print(paths)
for path in paths:
    with open(os.path.join(path, "trainer_state.json")) as f:
        info = json.load(f)
    value = info["log_history"][-1][field]
    map[path] = value
    smallest = min(value, smallest)

map = dict(sorted(map.items(), key=lambda x: x[1]))
for k, v in map.items():
    print(k, math.ceil(total_steps * smallest / v))
