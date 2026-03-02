import torch


class RunningRewardNormalizer:
    def __init__(self, eps: float = 1e-8, momentum: float = 0.99):
        self.eps = eps
        self.momentum = momentum
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

        if name not in self.initialized:
            self.mean[name] = values.mean().detach()
            self.var[name] = values.var(unbiased=False).detach()
            self.initialized[name] = True
        else:
            batch_mean = values.mean().detach()
            batch_var = values.var(unbiased=False).detach()

            self.mean[name] = (
                self.momentum * self.mean[name]
                + (1 - self.momentum) * batch_mean
            )
            self.var[name] = (
                self.momentum * self.var[name]
                + (1 - self.momentum) * batch_var
            )

        std = torch.sqrt(self.var[name] + self.eps)
        normalized = (values - self.mean[name]) / std
        return torch.clamp(normalized, -5.0, 5.0)
