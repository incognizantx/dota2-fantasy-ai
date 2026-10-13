"""
Scoring logic for Dota 2 Fantasy emblems and banners.
Computes base values, tier bonuses, and trait synergies based on in-game compendium mechanics.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from fantasy_data import Color, Role, Trait, TIER_BONUSES, STAT_BASELINES


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


def calculate_emblem_multiplier(emblem: Emblem, trait_bonus: float) -> float:
    """
    Additive in-game multiplier:
    Total Multiplier = 1.0 (base 100%) + Tier_Bonus + Trait_Bonus
    Example: 100% base + 150% (Tier V) + 50% (Vampiric) = 300% (3.0x multiplier)
    """
    tier_bonus = TIER_BONUSES.get(emblem.tier, 0.10)
    total_mult = max(0.0, 1.0 + tier_bonus + trait_bonus)
    return total_mult


def calculate_emblem_score(role: Role, emblem: Emblem, trait_bonus: float) -> float:
    """
    Calculate single emblem fantasy points:
    Points = Base_Stat_Average * Total_Multiplier
    """
    base = STAT_BASELINES[role][emblem.color].get(emblem.stat, 0.0)
    multiplier = calculate_emblem_multiplier(emblem, trait_bonus)
    return base * multiplier


def calculate_banner_trait_bonuses(emblems: List[Emblem]) -> List[float]:
    """
    Computes trait bonus percentage for each emblem slot on a banner according to in-game mechanics:
    - Unique: +50% (+0.50) if all traits on the banner are unique.
    - Vampiric: +50% (+0.50) to self.
    - Benevolent: +20% (+0.20) when adjacent to another Benevolent, 0% standalone.
    - Friendly: +50% (+0.50) when at least 2 Friendly traits are on the banner.
    - Incorruptible: inverse tier scaling (+60% for Tier 1, +50% for Tier 2, etc.).
    """
    n = len(emblems)
    bonuses = [0.0] * n

    traits = [e.trait for e in emblems]

    # Pre-conditions
    friendly_count = traits.count(Trait.FRIENDLY)
    all_traits_unique = len(set(traits)) == n

    for i, emblem in enumerate(emblems):
        t = emblem.trait

        # 1. Unique: +50% when this is the only Unique trait on the banner
        if t == Trait.UNIQUE:
            bonuses[i] = 0.50 if traits.count(Trait.UNIQUE) == 1 else 0.0

        # 2. Vampiric: +50% to self
        elif t == Trait.VAMPIRIC:
            bonuses[i] = 0.50

        # 3. Benevolent: +20% if adjacent to another Benevolent, 0% otherwise
        elif t == Trait.BENEVOLENT:
            has_benevolent_neighbor = (
                (i > 0 and emblems[i - 1].trait == Trait.BENEVOLENT) or
                (i < n - 1 and emblems[i + 1].trait == Trait.BENEVOLENT)
            )
            bonuses[i] = 0.20 if has_benevolent_neighbor else 0.0

        # 4. Friendly: +50% if stacked (>= 2 on banner)
        elif t == Trait.FRIENDLY:
            bonuses[i] = 0.50 if friendly_count >= 2 else 0.0

        # 5. Incorruptible: inverse tier bonus (+60% on T1, +50% on T2, etc.)
        elif t == Trait.INCORRUPTIBLE:
            bonuses[i] = max(0.20, 0.70 - (emblem.tier * 0.10))

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
