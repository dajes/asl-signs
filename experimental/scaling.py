import numpy as np


def get_transfomer_parameters(n_layers, n_heads, kv_size, mlp_ratio):
    return n_layers * ((7 + 5 * kv_size + (mlp_ratio - 1) * 2 + (kv_size - 1) * (mlp_ratio - 1)) * n_heads + (
            kv_size - 7 + (kv_size - 1) * (mlp_ratio - 1)) * (n_heads - 1) + mlp_ratio * kv_size) * kv_size * n_heads


def get_head_parameters(n_heads, kv_size, n_features, maxlen, n_outputs):
    d_model = kv_size * n_heads
    return d_model * (maxlen + n_outputs + n_features + 6) + n_outputs


def get_total_parameters(n_layers, n_heads, kv_size, mlp_ratio, n_features, maxlen, n_outputs):
    return (get_transfomer_parameters(n_layers, n_heads, kv_size, mlp_ratio) +
            get_head_parameters(n_heads, kv_size, n_features, maxlen, n_outputs))


def assert_works():
    from module import Module
    n_features = 192
    n_outputs = [250]
    maxlen = 128

    for mlp_ratio in range(1, 4):
        for kv_size in range(1, 6):
            for n_layers in range(1, 3):
                for n_heads in range(1, 4):
                    d_model = kv_size * n_heads
                    module = Module(
                        '', 1, 1, 1, n_features, d_model, n_outputs, maxlen, 0, n_layers, n_heads, mlp_ratio,
                        'transformer', 0)
                    n_parameters = sum(p.numel() for p in module.model.parameters())
                    pred_parameter_count = get_total_parameters(n_layers, n_heads, kv_size, mlp_ratio, n_features,
                                                                maxlen, n_outputs[0])
                    assert pred_parameter_count == n_parameters


def get_best_params(target_parameters, n_features, maxlen, n_outputs, kv_size=64, mlp_ratio=4):
    n_parameters = 0
    max_layers = 0
    while n_parameters <= target_parameters:
        max_layers += 1
        n_parameters = get_total_parameters(max_layers, 1, kv_size, mlp_ratio, n_features, maxlen, n_outputs)

    n_parameters = 0
    max_heads = 0
    while n_parameters <= target_parameters:
        max_heads += 1
        n_parameters = get_total_parameters(1, max_heads, kv_size, mlp_ratio, n_features, maxlen, n_outputs)

    proximity = np.zeros((max_heads - 1, max_layers - 1))
    parameters = np.zeros((max_heads - 1, max_layers - 1), int)
    for n_layers in range(1, max_layers):
        for n_heads in range(1, max_heads):
            p = get_total_parameters(n_layers, n_heads, kv_size, mlp_ratio, n_features, maxlen, n_outputs)
            parameters[n_heads - 1, n_layers - 1] = p
            if p > target_parameters:
                break
            proximity[n_heads - 1, n_layers - 1] = p / target_parameters

    best_layers = proximity.argmax(1)[::-1]
    mask = np.ones(best_layers.shape, bool)
    mask[1:] = best_layers[1:] != best_layers[:-1]
    best_heads = np.arange(max_heads - 1, 0, -1)[mask]
    best_layers = best_layers[mask] + 1
    return best_heads, best_layers, parameters[best_heads - 1, best_layers - 1]


def get_best_scale(target_parameters, depth_to_width_ratio, n_features, maxlen, n_outputs, kv_size=64, mlp_ratio=4,
                   verbose=False):
    assert 0 <= depth_to_width_ratio <= 1
    best_heads, best_layers, best_parameters = get_best_params(
        target_parameters, n_features, maxlen, n_outputs, kv_size, mlp_ratio)

    index = int(round(depth_to_width_ratio * (len(best_heads) - 1) - 1E-8))
    heads = best_heads[index]
    layers = best_layers[index]
    parameters = best_parameters[index]
    if verbose:
        print(F'Chosen {index + 1}/{len(best_heads)}: {heads} heads, {layers} layers, {parameters:,} '
              F'({parameters/target_parameters:.1%}) parameters')
    return heads, layers


if __name__ == '__main__':
    get_best_scale(2e6, .52, 192, 256, 250, 64, 4, True)
