import torch


class CriticCalibrator:
    """
    Converts raw critic outputs into well-behaved continuous reward signals.
    """

    def __init__(self):
        # temperature scaling per critic
        self.temp = {
            "bind": 1.0,
            "safety": 1.0,
            "reaction": 1.0,
            "vision": 1.0,
        }
        self.target_std = 1.0
        self.adapt_rate = 0.01

    def adapt_temperature(self, name: str, values: torch.Tensor) -> None:
        """
        Adjust temperature so critic output variance stays near target_std.
        """
        if values.numel() <= 1:
            return
        current_std = values.std(unbiased=False).detach()
        if current_std > 0:
            ratio = current_std / self.target_std
            self.temp[name] *= float(1 + self.adapt_rate * (ratio - 1))
            self.temp[name] = float(torch.clamp(torch.tensor(self.temp[name]), 0.1, 10.0))

    def calibrate_binding(self, raw: torch.Tensor) -> torch.Tensor:
        # Assume regression output where lower affinity proxy can be better.
        scaled = -raw / self.temp["bind"]
        self.adapt_temperature("bind", scaled)
        return torch.tanh(scaled)

    def calibrate_safety(self, logits: torch.Tensor) -> torch.Tensor:
        # Positive safety risk should be penalized.
        scaled = -logits / self.temp["safety"]
        self.adapt_temperature("safety", scaled)
        return torch.tanh(scaled)

    def calibrate_reaction(self, logits: torch.Tensor) -> torch.Tensor:
        # Preserve continuous reaction signal (no thresholding).
        scaled = logits / self.temp["reaction"]
        self.adapt_temperature("reaction", scaled)
        return torch.tanh(scaled)

    def calibrate_vision(self, raw: torch.Tensor) -> torch.Tensor:
        scaled = raw / self.temp["vision"]
        self.adapt_temperature("vision", scaled)
        return torch.tanh(scaled)

    def apply_ood_dampening(self, logits: torch.Tensor, max_mag: float = 10.0) -> torch.Tensor:
        """
        Prevent extreme values from dominating reward aggregation.
        """
        return torch.clamp(logits, -max_mag, max_mag)

    def confidence_weight(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence weight in [0, 1].
        High entropy -> lower confidence for classification-style outputs.
        """
        if logits.dim() > 1 and logits.size(-1) > 1:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)
            max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float, device=logits.device))
            confidence = 1 - (entropy / (max_entropy + 1e-8))
        else:
            # Regression proxy: higher magnitude away from indecision -> higher confidence.
            confidence = torch.sigmoid(torch.abs(logits))
        return confidence.detach().float()

    def ood_penalty(self, logits: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
        """
        Penalize extreme logit magnitudes (OOD behavior proxy).
        """
        magnitude = torch.abs(logits)
        return torch.relu(magnitude - threshold)

    @staticmethod
    def ensemble_forward(models, model_input):
        """
        Utility for optional critic ensemble stabilization.
        Returns (mean, std) across model outputs.
        """
        outputs = [m(model_input) for m in models]
        stacked = torch.stack(outputs, dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)
