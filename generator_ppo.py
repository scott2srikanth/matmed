"""Token-level PPO on the existing generator, independent of coordinator PPO."""
import copy

import torch
from torch.distributions import Categorical


def token_distribution(model, inputs):
    logits, _ = model(inputs, key_padding_mask=inputs.eq(model.tokenizer.pad_idx))
    # Identical support at rollout and every PPO replay; no top-k truncation.
    logits = logits.clone()
    logits[..., [model.tokenizer.pad_idx, model.tokenizer.sos_idx,
                 model.tokenizer.unk_idx]] = -torch.inf
    logits[:, 0, model.tokenizer.eos_idx] = -torch.inf
    return Categorical(logits=logits)


@torch.no_grad()
def rollout(model, batch_size):
    model.eval()  # Disable dropout, not gradients in subsequent PPO replay.
    tok = model.tokenizer
    device = next(model.parameters()).device
    ids = torch.full((batch_size, 1), tok.sos_idx, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for _ in range(model.max_len - 1):
        next_ids = token_distribution(model, ids).sample()[:, -1]
        next_ids = torch.where(finished, tok.pad_idx, next_ids)
        ids = torch.cat([ids, next_ids[:, None]], dim=1)
        finished |= next_ids.eq(tok.eos_idx)
        if finished.all():
            break
    return ids


class GeneratorPPO:
    def __init__(self, model, lr=1e-5, clip=0.1, epochs=4,
                 entropy_coef=0.1, kl_coef=0.1, target_kl=0.02, sft_coef=0.2):
        self.model = model
        self.reference = copy.deepcopy(model).eval().requires_grad_(False)
        model.prepare_finetuning()
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        self.clip, self.epochs = clip, epochs
        self.entropy_coef, self.kl_coef = entropy_coef, kl_coef
        self.target_kl, self.sft_coef = target_kl, sft_coef

    def update(self, ids, rewards, replay_ids):
        self.model.eval()
        src, targets = ids[:, :-1], ids[:, 1:]
        mask = targets.ne(self.model.tokenizer.pad_idx)
        # PAD actions are never scored, but gather needs an in-support index.
        safe_targets = targets.masked_fill(~mask, self.model.tokenizer.char2idx['C'])
        if rewards.shape != (ids.size(0),) or not torch.isfinite(rewards).all():
            raise ValueError("Expected finite rewards with shape [batch]")
        with torch.no_grad():
            old = token_distribution(self.model, src)
            old_log = old.log_prob(safe_targets)
            reference = token_distribution(self.reference, src)
            advantages = rewards - rewards.mean()
            advantages = advantages / rewards.std(unbiased=False).clamp_min(1e-6)
        used_coef = self.kl_coef
        average = lambda x: x[mask].mean()

        def divergence(left, right):
            # Avoid 0 * (-inf - -inf) on the masked support.
            support = left.probs > 0
            left_log = left.logits.masked_fill(~support, 0)
            right_log = right.logits.masked_fill(~support, 0)
            return average((left.probs * (left_log - right_log)).sum(-1))

        for epoch in range(self.epochs):
            new = token_distribution(self.model, src)
            ratio = (new.log_prob(safe_targets) - old_log).exp()
            adv = advantages[:, None]
            policy_loss = -average(torch.minimum(ratio * adv,
                ratio.clamp(1 - self.clip, 1 + self.clip) * adv))
            prior_kl = divergence(new, reference)
            entropy = average(new.entropy())
            sft_loss = self.model.compute_loss(replay_ids[:, :-1], replay_ids[:, 1:])
            loss = policy_loss + used_coef * prior_kl - self.entropy_coef * entropy
            loss = loss + self.sft_coef * sft_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite generator PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0,
                                                       error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                updated = token_distribution(self.model, src)
                old_kl = divergence(old, updated)
                prior_kl = divergence(updated, reference)
            if old_kl > self.target_kl:
                break
        observed = float(prior_kl)
        if observed > self.target_kl:
            self.kl_coef *= 1.5
        elif observed < self.target_kl / 2:
            self.kl_coef *= 0.8
        self.kl_coef = min(10.0, max(1e-4, self.kl_coef))
        return dict(policy_loss=float(policy_loss.detach()), loss=float(loss.detach()),
                    prior_kl=observed, rollout_kl=float(old_kl),
                    entropy=float(average(updated.entropy())),
                    grad_norm=float(grad_norm), kl_coef_used=used_coef,
                    next_kl_coef=self.kl_coef, ppo_epochs=epoch + 1)
