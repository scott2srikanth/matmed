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
