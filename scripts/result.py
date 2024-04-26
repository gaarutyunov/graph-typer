import json
import pathlib

import pandas as pd

#%%
columns = ["Exact Match", "Up to Parametric Type"]
idx_itr = [
    [f"Top-{f}" for f in [1, 3, 5, 10]],
    [
        "GraphTyper",
        "Typilus",
        "Type4Py",
        "TypeWriter"
    ],
]
idx = pd.MultiIndex.from_product(idx_itr, names=["Top-n", "Model"])
#%%
data = []
#%%
model = "Final"
ctx = 1024

for top_n in [1, 3, 5, 10]:
    model_path = pathlib.Path(f"ckpts/{model} {ctx}")
    top_n_path = model_path / f"result_top_{top_n}.json"
    result = json.loads(top_n_path.read_text())

    exact_match = result["accuracy"]
    exact_match_utpt = result["accuracy_up_to_parametric_type"]
    data.append([exact_match, exact_match_utpt])
#%%
data.extend([
    [0.561, 0.583],
    [0.637, 0.673],
    [0.659, 0.704],
    [0.682, 0.732],
]) # TypeWriter

data.extend([
    [0.661, 0.742],
    [0.716, 0.798],
    [0.727, 0.809],
    [0.733, 0.815],
]) # Typilus

data.extend([
    [0.758, 0.806],
    [0.781, 0.838],
    [0.787, 0.847],
    [0.792, 0.854],
]) # Type4Py
#%%
df = pd.DataFrame(data, columns=columns, index=idx)
#%%
with open("../research/result.tex", mode="w") as f:
    print((df * 100).to_latex(
        float_format="{:.2f}".format,
    ), file=f)
