import torch


class RunningRewardNormalizer:
    def __init__(self, eps: float = 1e-8, momentum: float = 0.99, min_std: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.min_std = min_std
        self.mean = {}
        self.var = {}
        self.initialized = {}

    def normalize(self, name: str, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            name: Component name key (e.g., "bind", "safety").
            values: Tensor with shape [batch] (or scalar tensor).
        """
        values = values.float().view(-1)
        if values.numel() == 0:
            return values
        if not torch.isfinite(values).all():
            raise ValueError('Non-finite critic rewards')

        if name not in self.initialized:
            self.mean[name] = values.mean().detach()
            self.var[name] = (values.var(unbiased=False).detach() if values.numel() > 1
                              else values.new_tensor(1.0))
            self.initialized[name] = True
        else:
            batch_mean = values.mean().detach()
            batch_var = values.var(unbiased=False).detach()

            old_mean = self.mean[name]
            self.mean[name] = (
                self.momentum * self.mean[name]
                + (1 - self.momentum) * batch_mean
            )
            self.var[name] = (
                self.momentum * self.var[name]
                + (1 - self.momentum) * batch_var
                + self.momentum * (1 - self.momentum) * (old_mean - batch_mean).square()
            )

        std = torch.sqrt(self.var[name] + self.eps).clamp_min(self.min_std)
        normalized = (values - self.mean[name]) / std
        return torch.clamp(normalized, -5.0, 5.0)
