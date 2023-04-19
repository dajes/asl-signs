import torch


class Mixup:
    def __init__(self, n_coords, mixup_alpha=0.2, same_threshold=0.99, any_slices: bool = True,
                 transform_other: bool = False):
        self.n_coords = n_coords
        self.mixup_alpha = mixup_alpha
        self.same_threshold = same_threshold
        self.any_slices = any_slices
        self.transform_other = transform_other
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
        lam = self.distribution.sample((x.shape[0], *[1] * (len(x.shape) - 1))).to(x.device)
        lam = torch.maximum(lam, 1 - lam)
        index = torch.randperm(x.size(0), device=x.device)

        if self.any_slices:
            starts = attn_mask.float().argmax(dim=1)
            ends = attn_mask.shape[1] - attn_mask.float().flip(dims=(1,)).argmax(dim=1)
            mixed_x = x.clone()
            mixed_attn = torch.zeros_like(attn_mask)
            for i, idx in enumerate(index):
                lambd = lam[i]
                if lambd > self.same_threshold:
                    mixed_attn[i] = attn_mask[i]
                    continue
                length = ends[i] - starts[i]
                length2 = ends[idx] - starts[idx]

                if length > length2:
                    start = torch.randint(starts[i], ends[i] - length2, (1,))[0]
                    current = mixed_x[i, start:start + length2]
                    mixed_x[i, start:start + length2] = lambd * current + (1 - lambd) * self.other_transform(current, x[idx, :length2])
                    mixed_attn[i, start:start + length2] = True
                elif length < length2:
                    start = torch.randint(starts[idx], ends[idx] - length, (1,))[0]
                    current = mixed_x[i, starts[i]:ends[i]]
                    mixed_x[i, starts[i]:ends[i]] = current * lambd + (1 - lambd) * self.other_transform(current, x[idx, start:start + length])
                    mixed_attn[i, starts[i]:ends[i]] = True
                else:
                    current = mixed_x[i, starts[i]:ends[i]]
                    mixed_x[i, starts[i]:ends[i]] = current * lambd + (1 - lambd) * self.other_transform(current, x[idx, starts[idx]:ends[idx]])
                    mixed_attn[i, starts[i]:ends[i]] = True
        else:
            mixed_x = lam * x + (1 - lam) * x[index, :]
            mixed_attn = attn_mask & (attn_mask[index] | (lam.squeeze(1) > self.same_threshold))

        return mixed_x, mixed_attn, lam, index

    @staticmethod
    def apply_criterion(criterion, pred, y, ds_nums, lam, index, reduction=None):
        if isinstance(lam, int) and lam == 1:
            return criterion(pred, y, ds_nums)
        if isinstance(lam, torch.Tensor):
            lam = lam.squeeze()
        init_reduction = criterion.reduction
        if reduction is None:
            reduction = init_reduction
        criterion.reduction = 'none'
        loss = lam * criterion(pred, y, ds_nums) + (1 - lam) * criterion(pred, y[index], ds_nums[index])
        criterion.reduction = init_reduction
        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()

        return loss
