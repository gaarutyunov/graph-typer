import pathlib

import numpy as np
import pandas as pd
import json
#%%
data = []

for top_n in [1, 3, 5]:
    for model in [
        "Ablated 512",
        "Base 512",
        "pldi2020_small-graph_coder_encoder_base",
        "pldi2020-graph_coder_autoencoder_base-120524",
        "pldi2020_small_1024-graph_coder_encoder_base",
        "pldi2020_small-graph_coder_encoder_big"
    ]:
        model_path = pathlib.Path(f"ckpts/{model}")
        top_n_path = model_path / f"result_top_{top_n}.json"
        result = json.loads(top_n_path.read_text())

        data.append([result["accuracy"], result["accuracy_up_to_parametric_type"]])
#%%
idx_itr = [
    [f"Top-{f}" for f in [1, 3, 5]],
    [
        "Plane Transformer",
        "+ Node \& Type Identifiers",
        "+ Type Annotations",
        "+ Decoder (Autoencoder)",
        "or Longer Context",
        "or More Parameters"
    ],
]
idx = pd.MultiIndex.from_product(idx_itr, names=["Top-n", "Model"])

df = pd.DataFrame(data, index=idx, columns=["Exact", "Up to Parametric Type"])
#%%
def highlight_max(data):
    attr = "font-weight:bold;"

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]

    is_max = data.groupby(level=0).transform('max') == data

    return pd.DataFrame(np.where(is_max, attr, ''),
                        index=data.index, columns=data.columns)


with open("../research/tables/ablation.tex", mode="w") as f:
    print((df * 100).style.apply(highlight_max, axis=None).format(precision=2).to_latex(
        convert_css=True,
        hrules=True,
        clines="skip-last;data",
    ), file=f)
