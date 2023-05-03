import numpy as np
import torch


class Mixup:
    def __init__(self, n_coords, mixup_alpha=0.2, same_threshold=0.999, any_slices: bool = True,
                 transform_other: bool = False, instances=2):
        assert 0 <= mixup_alpha <= 1
        assert 0 <= same_threshold <= 1
        assert instances > 1
        self.n_coords = n_coords
        self.mixup_alpha = mixup_alpha
        self.same_threshold = same_threshold
        self.any_slices = any_slices
        self.transform_other = transform_other
        self.instances = instances
        if self.mixup_alpha:
            self.distribution = torch.distributions.beta.Beta(mixup_alpha, mixup_alpha)

    def __call__(self, x, attn_mask, training):
        if self.mixup_alpha and training:
            return self.batch_mixup(x, attn_mask)
        return x, attn_mask, 1, None

    def other_transform(self, current, other):
        if not self.transform_other:
            return other
        y = current.reshape(-1, self.n_coords)
        x = other.reshape(-1, self.n_coords)
        m = (x != 0).any(1) & (y != 0).any(1)

        weights = torch.linalg.lstsq(x[m], y[m]).solution

        return (x @ weights).reshape(other.shape)

    @torch.jit.export
    def batch_mixup(self, x, attn_mask):
        lam = self.distribution.sample((self.instances - 1, x.shape[0], *[1] * (len(x.shape) - 1))).to(x.device)
        lam = torch.maximum(lam, 1 - lam)

        index = np.stack([np.random.permutation(x.size(0)) for _ in range(x.size(0))], 1)
        for j in range(len(lam)):
            for k in range(index.shape[1]):
                if index[j][k] == k:
                    index[j][k] = index[-1][k]
        index = torch.from_numpy(index[:len(lam)]).to(x.device)

        orig_x = x
        x = x.clone()

        for lam_, index_ in zip(lam, index):
            if self.any_slices:
                starts = attn_mask.float().argmax(dim=1)
                ends = attn_mask.shape[1] - attn_mask.float().flip(dims=(1,)).argmax(dim=1)
                mixed_attn = torch.zeros_like(attn_mask)
                for i, idx in enumerate(index_):
                    lambd = lam_[i]
                    if lambd > self.same_threshold:
                        mixed_attn[i] = attn_mask[i]
                        continue
                    length = ends[i] - starts[i]
                    length2 = ends[idx] - starts[idx]

                    if length > length2:
                        start = torch.randint(starts[i], ends[i] - length2, (1,))[0]
                        current = x[i, start:start + length2]
                        x[i, start:start + length2] = lambd * current + (1 - lambd) * self.other_transform(
                            current, orig_x[idx, :length2])
                        mixed_attn[i, start:start + length2] = True
                    elif length < length2:
                        start = torch.randint(starts[idx], ends[idx] - length, (1,))[0]
                        current = x[i, starts[i]:ends[i]]
                        x[i, starts[i]:ends[i]] = current * lambd + (1 - lambd) * self.other_transform(current,
                                                                                                       orig_x[idx,
                                                                                                       start:start + length])
                        mixed_attn[i, starts[i]:ends[i]] = True
                    else:
                        current = x[i, starts[i]:ends[i]]
                        x[i, starts[i]:ends[i]] = current * lambd + (1 - lambd) * self.other_transform(current,
                                                                                                       orig_x[idx,
                                                                                                       starts[
                                                                                                           idx]:
                                                                                                       ends[idx]])
                        mixed_attn[i, starts[i]:ends[i]] = True
            else:
                x = lam_ * x + (1 - lam_) * orig_x[index_, :]
                mixed_attn = attn_mask & (attn_mask[index_] | (lam_.squeeze(1) > self.same_threshold))
            attn_mask = mixed_attn

        return x, attn_mask, lam, index

    @staticmethod
    def apply_criterion(criterion, pred, y, ds_nums, lam, index, reduction=None):
        if isinstance(lam, int) and lam == 1:
            return criterion(pred, y, ds_nums)
        if isinstance(lam, torch.Tensor):
            lam = lam.squeeze(-1).squeeze(-1)
        init_reduction = criterion.reduction
        if reduction is None:
            reduction = init_reduction
        criterion.reduction = 'none'
        loss = criterion(pred, y, ds_nums)
        for lam_, index_ in zip(lam, index):
            loss = lam_ * loss + (1 - lam_) * criterion(pred, y[index_], ds_nums[index_])

        criterion.reduction = init_reduction
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()

        return loss


def visualize_density(mixup_alpha=1., instances=3, samples=100_000, resolution=50):
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    points = np.linspace(0, 2 * np.pi, instances + 1)[1:]
    x = np.cos(points)
    y = np.sin(points)
    xys = np.stack([x, y], 1)
    sampled = []
    distribution = torch.distributions.beta.Beta(mixup_alpha, mixup_alpha)
    lam = distribution.sample((samples, instances - 1,))
    lam = torch.maximum(lam, 1 - lam).numpy()
    for i in tqdm(range(samples)):
        sample = xys.copy()
        index = np.stack([np.random.permutation(len(xys)) for _ in range(len(xys))], 1)
        for j in range(len(lam[i])):
            for k in range(index.shape[1]):
                if index[j][k] == k:
                    index[j][k] = index[-1][k]
        for lam_, index_ in zip(lam[i], index):
            sample = lam_ * sample + (1 - lam_) * xys[index_]
        sampled.append(sample)
    sampled = np.concatenate(sampled, 0)
    heightmap = np.zeros((resolution, resolution), dtype=np.int64)
    for x, y in sampled:
        x = max(min(int((x + 1) / 2 * resolution), resolution - 1), 0)
        y = max(min(int((y + 1) / 2 * resolution), resolution - 1), 0)
        heightmap[x, y] += 1

    rows = np.where(np.any(heightmap, axis=1))[0]
    assert len(rows)
    cols = np.where(np.any(heightmap, axis=0))[0]
    assert len(cols)
    x1 = rows[0]
    x2 = rows[-1] + 1
    y1 = cols[0]
    y2 = cols[-1] + 1
    x = (x1 + x2) // 2
    y = (y1 + y2) // 2
    s = max(x2 - x1, y2 - y1)
    x1 = max(x - s // 2, 0)
    x2 = min(x + s // 2 + 1, resolution)
    y1 = max(y - s // 2, 0)
    y2 = min(y + s // 2 + 1, resolution)

    plt.figure(figsize=(max((y2 - y1) // 2, 3), max((x2 - x1) // 2, 3)))
    plt.imshow(np.log(heightmap[x1:x2, y1:y2] + 1))
    for i in range(x1, x2):
        for j in range(y1, y2):
            plt.text(j - y1, i - x1, heightmap[i, j], ha="center", va="center", color="w")

    plt.scatter(*((xys.T[::-1] + 1) * .5 * (resolution - 0) - .5 - [[y1], [x1]]), color='r', marker='x')

    plt.title(f'mixup_alpha={mixup_alpha}, instances={instances}, samples={samples}')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)
    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    for n_instances in range(2, 5):
        visualize_density(1, n_instances)
