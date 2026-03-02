import torch
import torch.nn as nn


class ModelWithTemperature(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature.clamp(min=1e-4)

    def set_temperature(self, val_loader, device):
        self.to(device)
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def _closure():
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                total_loss = total_loss + nll_criterion(
                    logits / self.temperature.clamp(min=1e-4),
                    y,
                )
            total_loss.backward()
            return total_loss

        optimizer.step(_closure)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self
