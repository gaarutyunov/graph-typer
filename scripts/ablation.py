import pathlib
import pandas as pd
import json
#%%
data = []

for top_n in [1, 3, 5, 10]:
    for model in [
        "Ablated",
        "Base",
        "Deep",
        "Big",
        "Final"
    ]:
        em = []
        em_utpt = []
        for i, ctx in enumerate([512, 1024]):
            model_path = pathlib.Path(f"ckpts/{model} {ctx}")
            top_n_path = model_path / f"result_top_{top_n}.json"
            result = json.loads(top_n_path.read_text())

            exact_match = result["accuracy"]
            exact_match_utpt = result["accuracy_up_to_parametric_type"]
            em.append(exact_match)
            em_utpt.append(exact_match_utpt)
        em.extend(em_utpt)
        data.append(em)
#%%
idx_itr = [
    [f"Top-{f}" for f in [1, 3, 5, 10]],
    [
        "Ablated (51 mln)",
        "Base (51 mln)",
        "Deep (214 mln)",
        "Big (331 mln)",
        "Final (432 mln)"
    ],
]
idx = pd.MultiIndex.from_product(idx_itr, names=["Top-n", "Model"])

col_iter = [["Exact", "Up to Parametric Type"], ["512", "1024"]]
col = pd.MultiIndex.from_product(col_iter, names=["\% Match", "Context Length"])

df = pd.DataFrame(data, index=idx, columns=col)
#%%
with open("../research/ablation.tex", mode="w") as f:
    print((df * 100).to_latex(
        float_format="{:.2f}".format,
    ), file=f)
