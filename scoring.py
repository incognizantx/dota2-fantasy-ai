"""
Scoring logic for Dota 2 Fantasy emblems and banners.
Computes base values, tier multipliers, and trait synergies.
"""

from dataclasses import dataclass
from typing import List, Dict
from fantasy_data import Color, Role, Trait, TIER_MULTIPLIERS, STAT_BASELINES


@dataclass
class Emblem:
    color: Color
    stat: str
    tier: int  # 1 to 5
    trait: Trait

    def copy(self) -> "Emblem":
        return Emblem(
            color=self.color,
            stat=self.stat,
            tier=self.tier,
            trait=self.trait,
        )


def calculate_emblem_score(role: Role, emblem: Emblem, trait_bonus: float) -> float:
    """
    Calculate single emblem score:
    Score = Base_Stat * Tier_Multiplier * (1.0 + Trait_Bonus)
    """
    base = STAT_BASELINES[role][emblem.color].get(emblem.stat, 0.0)
    tier_mult = TIER_MULTIPLIERS.get(emblem.tier, 1.0)
    # Total multiplier cannot drop below 0
    total_mult = max(0.0, 1.0 + trait_bonus)
    return base * tier_mult * total_mult


def calculate_banner_trait_bonuses(emblems: List[Emblem]) -> List[float]:
    """
    Computes trait bonus percentage (e.g. +0.5 for +50%) for each emblem slot on a banner.
    Supported traits:
    - Friendly: +50% if banner has at least 2 Friendly traits.
    - Vampiric: +50% to self, -25% to immediate neighbors (left & right).
    - Benevolent: +20% to immediate neighbors (left & right).
    - Unique: +30% to self if all traits on banner are unique.
    - Fractal: +50% to self if all tiers on banner are unique.
    """
    n = len(emblems)
    bonuses = [0.0] * n

    traits = [e.trait for e in emblems]
    tiers = [e.tier for e in emblems]

    # Pre-conditions
    friendly_count = traits.count(Trait.FRIENDLY)
    all_traits_unique = len(set(traits)) == n
    all_tiers_unique = len(set(tiers)) == n

    for i, emblem in enumerate(emblems):
        t = emblem.trait

        # 1. Friendly
        if t == Trait.FRIENDLY and friendly_count >= 2:
            bonuses[i] += 0.50

        # 2. Vampiric
        elif t == Trait.VAMPIRIC:
            bonuses[i] += 0.50
            if i > 0:
                bonuses[i - 1] -= 0.25
            if i < n - 1:
                bonuses[i + 1] -= 0.25

        # 3. Benevolent
        elif t == Trait.BENEVOLENT:
            if i > 0:
                bonuses[i - 1] += 0.20
            if i < n - 1:
                bonuses[i + 1] += 0.20

        # 4. Unique
        elif t == Trait.UNIQUE and all_traits_unique:
            bonuses[i] += 0.30

        # 5. Fractal
        elif t == Trait.FRACTAL and all_tiers_unique:
            bonuses[i] += 0.50

    return bonuses


def calculate_banner_score(role: Role, emblems: List[Emblem]) -> float:
    """Calculate the total score of a single role banner."""
    bonuses = calculate_banner_trait_bonuses(emblems)
    total = 0.0
    for emblem, bonus in zip(emblems, bonuses):
        total += calculate_emblem_score(role, emblem, bonus)
    return total


def calculate_total_fantasy_score(roster: Dict[Role, List[Emblem]]) -> float:
    """Calculate total fantasy score across all active role banners."""
    return sum(calculate_banner_score(role, emblems) for role, emblems in roster.items())
