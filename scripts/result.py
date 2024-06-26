import json
import pathlib

import numpy as np
import pandas as pd

#%%
columns = ["Exact Match", "Up to Parametric Type"]
idx_itr = [
    [f"Top-{f}" for f in [1, 3, 5]],
    [
        "GraphTyper",
        "TypeBERT",
        "TypeWriter",
        "Typilus",
        "Type4Py",
        "TypeGen"
    ],
]
idx = pd.MultiIndex.from_product(idx_itr, names=["Top-n", "Model"])
#%%
data = []
#%%
model = "Final"
ctx = 1024

data_ = []

for top_n in [1, 3, 5]:
    model_path = pathlib.Path(f"ckpts/{model} {ctx}")
    top_n_path = model_path / f"result_top_{top_n}.json"
    result = json.loads(top_n_path.read_text())

    exact_match = result["accuracy"]
    exact_match_utpt = result["accuracy_up_to_parametric_type"]
    data_.append([exact_match, exact_match_utpt])

data.append(data_)
#%%
data.append([
    [0.454, 0.481],
    [0.514, 0.535],
    [0.541, 0.565]
])  # TypeBERT

data.append([
    [0.561, 0.583],
    [0.637, 0.673],
    [0.659, 0.704],
])  # TypeWriter

data.append([
    [0.661, 0.742],
    [0.716, 0.798],
    [0.727, 0.809],
])  # Typilus

data.append([
    [0.758, 0.806],
    [0.781, 0.838],
    [0.787, 0.847],
])  # Type4Py

data.append([
    [0.792, 0.873],
    [0.856, 0.910],
    [0.870, 0.917],
])  # TypeGen
#%%
df = pd.DataFrame([data[j][i] for i in range(3) for j in range(len(data))], columns=columns, index=idx)
#%%
def highlight_max(data):
    attr = "font-weight:bold;"

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]

    is_max = data.groupby(level=0).transform('max') == data

    return pd.DataFrame(np.where(is_max, attr, ''),
                        index=data.index, columns=data.columns)


with open("../research/tables/result.tex", mode="w") as f:
    print((df * 100).style.apply(highlight_max, axis=None).format(precision=2).to_latex(
        convert_css=True,
        hrules=True,
        clines="skip-last;data",
    ), file=f)

#%%
columns_landscape = [
    [f"Top-{f}" for f in [1, 3, 5]],
    ["EM", "UTPT"]
]
columns_landscape_idx = pd.MultiIndex.from_product(columns_landscape, names=["Top-n", "Metric"])
idx_landscape = [
    "GraphTyper",
    "TypeBERT",
    "TypeWriter",
    "Typilus",
    "Type4Py",
    "TypeGen"
]


def flatten(arr):
    res = []
    for a in arr:
        res.extend(a)
    return res


df_landscape = pd.DataFrame([flatten(model) for model in data], columns=columns_landscape_idx, index=idx_landscape)
#%%
with open("../research/tables/result-presentation.tex", mode="w") as f:
    print((df_landscape * 100).style.highlight_max(color=None,props="font-weight:bold;").format(precision=2).to_latex(
        convert_css=True,
        hrules=True,
        clines="skip-last;data",
    ), file=f)
