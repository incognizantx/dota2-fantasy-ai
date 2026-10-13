"""
Dota 2 Fantasy Optimizer - Constants and Game Data.
Contains stats baseline values, tier multipliers, traits, and available operations.
"""

from enum import Enum
from typing import Dict, List, Tuple


class Color(str, Enum):
    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"


class Role(str, Enum):
    CORE = "CORE"
    MID = "MID"
    SUPPORT = "SUPPORT"


class Trait(str, Enum):
    FRACTAL = "Fractal"
    FRIENDLY = "Friendly"
    VAMPIRIC = "Vampiric"
    UNIQUE = "Unique"
    BENEVOLENT = "Benevolent"


ALL_TRAITS = [Trait.FRACTAL, Trait.FRIENDLY, Trait.VAMPIRIC, Trait.UNIQUE, Trait.BENEVOLENT]

# Tier multipliers from game rules:
# Tier 1: +10% (1.10)
# Tier 2: +30% (1.30)
# Tier 3: +60% (1.60)
# Tier 4: +100% (2.00)
# Tier 5: +150% (2.50)
TIER_MULTIPLIERS: Dict[int, float] = {
    1: 1.10,
    2: 1.30,
    3: 1.60,
    4: 2.00,
    5: 2.50,
}

# Banner layout by stage
STAGE_1_LAYOUT: Dict[Role, List[Color]] = {
    Role.CORE: [Color.RED, Color.GREEN, Color.RED],
    Role.MID: [Color.RED, Color.BLUE, Color.GREEN],
    Role.SUPPORT: [Color.BLUE, Color.GREEN, Color.BLUE],
}

STAGE_2_LAYOUT: Dict[Role, List[Color]] = {
    Role.CORE: [Color.RED, Color.GREEN, Color.RED, Color.GREEN, Color.RED],
    Role.MID: [Color.RED, Color.BLUE, Color.GREEN, Color.RED, Color.GREEN],
    Role.SUPPORT: [Color.BLUE, Color.GREEN, Color.BLUE, Color.GREEN, Color.BLUE],
}

# Baseline Fantasy Points from guide (Reddit TI guide tables)
# Maps: Role -> Color -> Stat Name -> Average Score
STAT_BASELINES: Dict[Role, Dict[Color, Dict[str, float]]] = {
    Role.CORE: {
        Color.RED: {
            "Creeps": 1438.0,
            "GPM": 1411.0,
            "Deaths": 1282.0,
            "Towers": 1013.0,
            "Kills": 857.0,
            "Madstones": 701.0,
        },
        Color.GREEN: {
            "Teamfight": 1305.0,
            "Roshan": 778.0,
            "Tormentor": 761.0,
            "Stuns": 194.0,
            "First Blood": 188.0,
            "Courier": 162.0,
        },
        Color.BLUE: {
            # Core does not have blue in standard stage 1/2, fallback default
            "Wards": 1032.0,
            "Smokes": 938.0,
            "Camps stacked": 889.0,
            "Watchers": 842.0,
            "Lotuses": 615.0,
            "Runes": 409.0,
        },
    },
    Role.MID: {
        Color.RED: {
            "Deaths": 1232.0,
            "GPM": 1206.0,
            "Creeps": 1134.0,
            "Kills": 775.0,
            "Madstones": 467.0,
            "Towers": 353.0,
        },
        Color.GREEN: {
            "Teamfight": 1489.0,
            "Tormentor": 446.0,
            "Stuns": 337.0,
            "Roshan": 281.0,
            "Courier": 216.0,
            "First Blood": 123.0,
        },
        Color.BLUE: {
            # Mid blue stats
            "Wards": 1032.0,
            "Smokes": 938.0,
            "Camps stacked": 889.0,
            "Watchers": 842.0,
            "Lotuses": 615.0,
            "Runes": 409.0,
        },
    },
    Role.SUPPORT: {
        Color.RED: {
            # Support fallback if red added
            "Deaths": 1232.0,
            "GPM": 800.0,
            "Creeps": 400.0,
            "Kills": 600.0,
            "Madstones": 400.0,
            "Towers": 300.0,
        },
        Color.GREEN: {
            "Teamfight": 1415.0,
            "Tormentor": 420.0,
            "Stuns": 389.0,
            "Courier": 294.0,
            "First Blood": 207.0,
            "Roshan": 50.0,
        },
        Color.BLUE: {
            "Wards": 1032.0,
            "Smokes": 938.0,
            "Camps stacked": 889.0,
            "Watchers": 842.0,
            "Lotuses": 615.0,
            "Runes": 409.0,
        },
    },
}

# The 20 Operations available in the Compendium
OPERATIONS = [
    # Green operations (6)
    "reroll_stat_green_all",
    "reroll_quality_green_all",
    "reroll_trait_green_all",
    "reroll_stat_green_first",
    "reroll_stat_green_last",
    "reroll_stat_green_random",
    # Red operations (6)
    "reroll_stat_red_all",
    "reroll_quality_red_all",
    "reroll_trait_red_all",
    "reroll_quality_red_first",
    "reroll_quality_red_last",
    "reroll_quality_red_random",
    # Blue operations (6)
    "reroll_stat_blue_all",
    "reroll_quality_blue_all",
    "reroll_trait_blue_all",
    "reroll_trait_blue_first",
    "reroll_trait_blue_last",
    "reroll_trait_blue_random",
    # Other wildcard operations (2)
    "increase_one_quality_random",
    "increase_two_reduce_one_quality_random",
]

OP_TO_ID = {op: i for i, op in enumerate(OPERATIONS)}
ID_TO_OP = {i: op for i, op in enumerate(OPERATIONS)}
