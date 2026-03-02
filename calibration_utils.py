import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)


def _forward_with_optional_unpack(model, x):
    if isinstance(x, (tuple, list)):
        return model(*x)
    return model(x)


def calibrate_model(model, val_loader, device):
    scaler = TemperatureScaler().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    model.eval()
    has_batches = False

    def _closure():
        nonlocal has_batches
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for x, y in val_loader:
            has_batches = True
            if isinstance(x, (tuple, list)):
                x = tuple(v.to(device) if torch.is_tensor(v) else v for v in x)
            elif torch.is_tensor(x):
                x = x.to(device)
            y = y.to(device)

            logits = _forward_with_optional_unpack(model, x)
            total_loss = total_loss + criterion(scaler(logits), y)

        if not has_batches:
            # No validation batches -> keep default temperature=1.0
            return total_loss
        total_loss.backward()
        return total_loss

    optimizer.step(_closure)
    return scaler
