import numpy as np
import pandas as pd

from experimental.scaling import get_best_params, get_total_parameters

sizes = pd.read_csv('chinchilla_sizes.csv')

n_features = 192
maxlen = 256
n_outputs = 250

scalers = []

for row in sizes.itertuples():
    params = row[1] * 1e6
    pred_params = get_total_parameters(
        n_layers=row.n_layers,
        n_heads=row.n_heads,
        kv_size=row.kv_size,
        mlp_ratio=row.ffw_size / row.d_model,
        n_features=n_features,
        maxlen=maxlen,
        n_outputs=n_outputs
    )
    best_heads, best_layers, _ = get_best_params(
        pred_params, n_features, maxlen, n_outputs, row.kv_size, row.ffw_size / row.d_model)

    hed_loc = np.argmin(np.abs(row.n_heads - best_heads))
    lay_loc = np.argmin(np.abs(row.n_layers - best_layers))

    scaling = (hed_loc / (best_layers.shape[0] - 1), lay_loc / (best_heads.shape[0] - 1))
    print(f'{row[1]}M {hed_loc + 1}/{best_heads.shape[0]} {lay_loc + 1}/{best_layers.shape[0]}')

    scalers.append(scaling)

scalers = np.array(scalers)
unique_scales, counts = np.unique(scalers, return_counts=True)

print(f'Mean depth/width scale: {scalers.mean():.4f}')
print(f'Median depth/width scale: {np.median(scalers):.4f}')
print(f'Mode depth/width scale: {unique_scales[np.argmax(counts)]:.4f}')
