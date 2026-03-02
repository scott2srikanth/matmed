"""
reward.py — MATMED Reward Function
====================================
Defines the composite reward signal used to guide the policy agent.

Scientific rationale:
  A drug candidate must score well on multiple competing objectives:
    - High binding affinity (efficacy)
    - Synthetic feasibility (yield)
    - Safe ADMET profile (safety)
    - Low toxicity

  The reward combines these with configurable coefficients so researchers
  can trade-off between objectives without touching agent code.

Reward formula:
  R = alpha * binding_score
    + beta  * yield_score
    + gamma * admet_score
    - delta * toxicity

All inputs are assumed to be in [0, 1] after normalisation.
"""

from dataclasses import dataclass, field
from typing import Dict
import torch

from reward_normalizer import RunningRewardNormalizer
from critic_calibration import CriticCalibrator


@dataclass
class RewardConfig:
    """
    Configurable coefficients for the MATMED reward function.

    Attributes:
        alpha:  Weight for binding affinity score (higher → prefer tight binders).
        beta:   Weight for synthetic yield score (higher → prefer easy synthesis).
        gamma:  Weight for ADMET score (higher → prefer drug-like molecules).
        delta:  Penalty coefficient for toxicity (higher → harder penalty).
        clip_min: Minimum clipped reward value.
        clip_max: Maximum clipped reward value.
    """
    alpha: float = 1.0   # binding affinity
    beta: float  = 0.5   # synthetic feasibility
    gamma: float = 0.5   # ADMET (druglikeness)
    delta: float = 1.0   # toxicity penalty
    clip_min: float = -5.0
    clip_max: float = 5.0


class RewardFunction:
    """
    Stateless reward calculator for MATMED.

    Usage::

        reward_fn = RewardFunction(RewardConfig(alpha=1.2, delta=1.5))
        r = reward_fn.compute(
            binding_score=0.8,
            yield_score=0.6,
            admet_score=0.7,
            toxicity=0.1,
        )
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        binding_score: float,
        yield_score: float,
        admet_score: float,
        toxicity: float,
    ) -> float:
        """
        Compute the scalar reward for a single molecule.

        Args:
            binding_score: Predicted binding affinity in [0, 1].
                           Higher is better (tighter binding).
            yield_score:   Predicted synthetic yield in [0, 1].
                           Higher means the reaction is more feasible.
            admet_score:   Overall ADMET quality score in [0, 1].
                           Higher means better druglikeness / safety profile.
            toxicity:      Predicted toxicity probability in [0, 1].
                           Higher means more toxic → subtracted as penalty.

        Returns:
            Scalar reward value (clipped to [clip_min, clip_max]).
        """
        c = self.config
        reward = (
            c.alpha * binding_score
            + c.beta  * yield_score
            + c.gamma * admet_score
            - c.delta * toxicity
        )
        return float(max(c.clip_min, min(c.clip_max, reward)))

    def breakdown(
        self,
        binding_score: float,
        yield_score: float,
        admet_score: float,
        toxicity: float,
    ) -> Dict[str, float]:
        """
        Return a per-component breakdown of the reward (useful for logging).

        Returns:
            Dict with keys: 'binding_term', 'yield_term', 'admet_term',
            'toxicity_penalty', 'total_reward'.
        """
        c = self.config
        binding_term   = c.alpha * binding_score
        yield_term     = c.beta  * yield_score
        admet_term     = c.gamma * admet_score
        tox_penalty    = c.delta * toxicity
        total          = binding_term + yield_term + admet_term - tox_penalty
        total_clipped  = float(max(c.clip_min, min(c.clip_max, total)))

        return {
            'binding_term':    binding_term,
            'yield_term':      yield_term,
            'admet_term':      admet_term,
            'toxicity_penalty': tox_penalty,
            'total_reward':    total_clipped,
        }


class RewardAggregator:
    """
    Running-normalized multi-objective reward aggregator.

    Each component is normalized independently with a running mean/std, then
    combined with configurable weights.
    """
    def __init__(
        self,
        w_bind: float = 0.5,
        w_safety: float = 1.0,
        w_reaction: float = 1.0,
        w_vision: float = 0.5,
        momentum: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        self.normalizer = RunningRewardNormalizer(eps=eps, momentum=momentum)
        self.calibrator = CriticCalibrator()
        self.w_bind = w_bind
        self.w_safety = w_safety
        self.w_reaction = w_reaction
        self.w_vision = w_vision

    @staticmethod
    def _to_tensor(v) -> torch.Tensor:
        if isinstance(v, torch.Tensor):
            return v.float().view(-1)
        return torch.tensor([float(v)], dtype=torch.float)

    @staticmethod
    def _reduce_to_batch(v: torch.Tensor) -> torch.Tensor:
        if v.dim() <= 1:
            return v.view(-1)
        return v.view(v.size(0), -1).mean(dim=-1)

    def aggregate(
        self,
        r_bind_raw,
        r_safety_raw,
        r_reaction_raw,
        r_vision_raw,
        return_details: bool = False,
    ):
        r_bind_t = self._to_tensor(r_bind_raw)
        r_safety_t = self._to_tensor(r_safety_raw)
        r_reaction_t = self._to_tensor(r_reaction_raw)
        r_vision_t = self._to_tensor(r_vision_raw)

        # Keep all tensors on the same device as the first component.
        device = r_bind_t.device
        r_safety_t = r_safety_t.to(device)
        r_reaction_t = r_reaction_t.to(device)
        r_vision_t = r_vision_t.to(device)

        # Calibration: OOD dampening + critic-specific smooth transforms.
        r_bind = self.calibrator.calibrate_binding(
            self.calibrator.apply_ood_dampening(r_bind_t)
        )
        r_safety = self.calibrator.calibrate_safety(
            self.calibrator.apply_ood_dampening(r_safety_t)
        )
        r_reaction = self.calibrator.calibrate_reaction(
            self.calibrator.apply_ood_dampening(r_reaction_t)
        )
        r_vision = self.calibrator.calibrate_vision(
            self.calibrator.apply_ood_dampening(r_vision_t)
        )

        # Confidence weighting (downweight uncertain critic outputs).
        c_bind = self.calibrator.confidence_weight(r_bind_t)
        c_safety = self.calibrator.confidence_weight(r_safety_t)
        c_reaction = self.calibrator.confidence_weight(r_reaction_t)
        c_vision = self.calibrator.confidence_weight(r_vision_t)
        r_bind = r_bind * c_bind
        r_safety = r_safety * c_safety
        r_reaction = r_reaction * c_reaction
        r_vision = r_vision * c_vision

        # Running normalization to equalize critic contribution scales.
        r_bind_n = self.normalizer.normalize("bind", r_bind)
        r_safety_n = self.normalizer.normalize("safety", r_safety)
        r_reaction_n = self.normalizer.normalize("reaction", r_reaction)
        r_vision_n = self.normalizer.normalize("vision", r_vision)

        total_reward = (
            self.w_bind * r_bind_n
            + self.w_safety * r_safety_n
            + self.w_reaction * r_reaction_n
            + self.w_vision * r_vision_n
        )

        # OOD penalty using raw critic magnitudes (before dampening/calibration).
        ood_pen = (
            self._reduce_to_batch(self.calibrator.ood_penalty(r_bind_t))
            + self._reduce_to_batch(self.calibrator.ood_penalty(r_safety_t))
            + self._reduce_to_batch(self.calibrator.ood_penalty(r_reaction_t))
        )
        total_reward = total_reward - 0.1 * ood_pen

        if return_details:
            return total_reward, {
                'bind': r_bind,
                'safety': r_safety,
                'reaction': r_reaction,
                'vision': r_vision,
                'bind_n': r_bind_n,
                'safety_n': r_safety_n,
                'reaction_n': r_reaction_n,
                'vision_n': r_vision_n,
                'c_bind': c_bind,
                'c_safety': c_safety,
                'c_reaction': c_reaction,
                'c_vision': c_vision,
                'ood_pen': ood_pen,
            }
        return total_reward
