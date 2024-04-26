import pandas
import pandas as pd

#%%
columns = ["Model", ""]
idx = pandas.MultiIndex.from_product([["\% Match"], ["Exact", "Up to Parametric Type"]], names=columns)
#%%
data = [
    [0.41, 0.459],
    [0.546, 0.641]
]
#%%
df = pd.DataFrame(data, columns=idx, index=["GraphTyper", "Typilus"])
#%%
with open("../research/result.tex", mode="w") as f:
    print((df * 100).to_latex(
        float_format="{:.2f}".format,
    ), file=f)
