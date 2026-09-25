import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self, num_outputs=1):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(num_outputs))

    @property
    def temperature(self):
        return self.log_temperature.clamp(-5., 5.).exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def _forward_with_optional_unpack(model, x):
    if isinstance(x, (tuple, list)):
        return model(*x)
    return model(x)


def calibrate_model(model, val_loader, device):
    was_training = model.training
    logits_cache, labels_cache = [], []
    try:
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                if isinstance(x, (tuple, list)):
                    x = tuple(v.to(device) if torch.is_tensor(v) else v for v in x)
                elif torch.is_tensor(x):
                    x = x.to(device)
                logits_cache.append(_forward_with_optional_unpack(model, x).detach())
                labels_cache.append(y.to(device).float())
    finally:
        model.train(was_training)
    if not logits_cache:
        raise ValueError('Calibration requires nonempty held-out validation data')
    logits, labels = torch.cat(logits_cache), torch.cat(labels_cache)
    if logits.shape != labels.shape or not torch.isfinite(logits).all():
        raise ValueError('Expected finite logits matching validation label shape')
    mask = torch.isfinite(labels) & ((labels == 0) | (labels == 1))
    if not mask.any():
        raise ValueError('No observed binary validation labels')
    scaler = TemperatureScaler(logits.size(-1) if logits.ndim > 1 else 1).to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=.1, max_iter=50,
                                 line_search_fn='strong_wolfe')

    def _closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(scaler(logits)[mask], labels[mask])
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(_closure)
    return scaler
