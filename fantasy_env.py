"""
Gymnasium Environment for Dota 2 Fantasy Compendium Optimizer.
Simulates banner rerolls, action masking, and fantasy score progression.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fantasy_data import (
    Color,
    Role,
    Trait,
    ALL_TRAITS,
    STAGE_1_LAYOUT,
    STAGE_2_LAYOUT,
    STAT_BASELINES,
    OPERATIONS,
    OP_TO_ID,
    ID_TO_OP,
    TIER_MULTIPLIERS,
)
from scoring import (
    Emblem,
    calculate_banner_score,
    calculate_total_fantasy_score,
)

# Probability distribution for rolling tiers (Tier 1 is common, Tier 5 is rare)
TIER_PROBABILITIES = np.array([0.40, 0.30, 0.18, 0.09, 0.03], dtype=np.float32)
TIER_VALUES = np.array([1, 2, 3, 4, 5], dtype=np.int32)


class DotaFantasyEnv(gym.Env):
    """
    Gymnasium environment for Dota 2 Fantasy League.

    State:
    - 3 Banners (Core, Mid, Support).
    - Each emblem has (color, stat, tier, trait).
    - Remaining reroll tokens (starts at 40 in Stage 1).
    - 3 currently offered operations from the game pool.

    Action:
    - Discrete(9): (banner_idx * 3 + offer_idx)
      banner_idx in [0: Core, 1: Mid, 2: Support]
      offer_idx in [0, 1, 2]

    Action Masking:
    - Provides a boolean mask of shape (9,) disabling operations that do not affect
      the selected banner (e.g. blue reroll on a red/green Core banner).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage: int = 1,
        initial_tokens: int = 40,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.stage = stage
        self.initial_tokens = initial_tokens
        self.render_mode = render_mode

        self.roles = [Role.CORE, Role.MID, Role.SUPPORT]
        self.layout = STAGE_1_LAYOUT if stage == 1 else STAGE_2_LAYOUT
        self.num_emblems_per_banner = 3 if stage == 1 else 5

        # 9 discrete actions: 3 banners x 3 offered operations
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # Vectorized representation:
        # - For each of 3 banners:
        #   - For each emblem:
        #     - stat_idx (normalized: 0..5 / 5.0)
        #     - tier (normalized: 1..5 / 5.0)
        #     - trait_idx (normalized: 0..4 / 4.0)
        # - tokens_left (normalized: 0..40 / 40.0)
        # - 3 offered operations (normalized op_id: 0..19 / 19.0)
        # Total per emblem: 3 features. 3 banners * 3 (or 5) emblems * 3 = 27 (or 45)
        # + 1 (tokens) + 3 (offers) = 31 (or 49) features.
        obs_dim = (3 * self.num_emblems_per_banner * 3) + 1 + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # State storage
        self.tokens = self.initial_tokens
        self.roster: Dict[Role, List[Emblem]] = {}
        self.current_offers: List[str] = []
        self.current_score: float = 0.0

    def _sample_random_stat(self, role: Role, color: Color) -> str:
        available_stats = list(STAT_BASELINES[role][color].keys())
        idx = self.np_random.integers(0, len(available_stats))
        return available_stats[idx]

    def _sample_random_tier(self) -> int:
        return int(self.np_random.choice(TIER_VALUES, p=TIER_PROBABILITIES))

    def _sample_random_trait(self) -> Trait:
        idx = self.np_random.integers(0, len(ALL_TRAITS))
        return ALL_TRAITS[idx]

    def _init_roster(self):
        """Generates initial starter emblems for all 3 banners."""
        self.roster = {}
        for role in self.roles:
            emblems = []
            for color in self.layout[role]:
                stat = self._sample_random_stat(role, color)
                # Starter cards usually start at Tier 1 or Tier 2
                tier = int(self.np_random.choice([1, 2], p=[0.7, 0.3]))
                trait = self._sample_random_trait()
                emblems.append(Emblem(color=color, stat=stat, tier=tier, trait=trait))
            self.roster[role] = emblems

    def _sample_offers(self) -> List[str]:
        """Draw 3 random operations from the pool."""
        indices = self.np_random.integers(0, len(OPERATIONS), size=3)
        return [OPERATIONS[i] for i in indices]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.tokens = self.initial_tokens
        self._init_roster()
        self.current_offers = self._sample_offers()
        self.current_score = calculate_total_fantasy_score(self.roster)

        obs = self._get_obs()
        info = {
            "tokens": self.tokens,
            "score": self.current_score,
            "action_mask": self.action_masks(),
            "offers": list(self.current_offers),
        }
        return obs, info

    def action_masks(self) -> np.ndarray:
        """
        Calculates action mask array of shape (9,):
        mask[banner_idx * 3 + offer_idx] is True if valid, False if inactive.
        """
        mask = np.zeros(9, dtype=bool)
        for banner_idx, role in enumerate(self.roles):
            banner_emblems = self.roster[role]
            banner_colors = [e.color for e in banner_emblems]

            for offer_idx, op in enumerate(self.current_offers):
                act_id = banner_idx * 3 + offer_idx
                is_valid = self._is_op_valid_for_banner(op, banner_emblems, banner_colors)
                mask[act_id] = is_valid

        # Edge case: If all 9 options somehow masked out, allow all to prevent deadlocks
        if not np.any(mask):
            mask[:] = True
        return mask

    def _is_op_valid_for_banner(
        self, op: str, emblems: List[Emblem], colors: List[Color]
    ) -> bool:
        """Checks whether an operation can be meaningfully applied to a banner."""
        if "green" in op and Color.GREEN not in colors:
            return False
        if "red" in op and Color.RED not in colors:
            return False
        if "blue" in op and Color.BLUE not in colors:
            return False

        # If operation increases quality, at least one emblem must be < Tier 5
        if op == "increase_one_quality_random":
            if all(e.tier >= 5 for e in emblems):
                return False

        if op == "increase_two_reduce_one_quality_random":
            # Needs at least 2 emblems that can increase or at least one to reduce
            if all(e.tier >= 5 for e in emblems):
                return False

        return True

    def _apply_operation(self, role: Role, op: str):
        """Executes the selected operation on the target banner."""
        emblems = self.roster[role]
        color_indices = {
            Color.RED: [i for i, e in enumerate(emblems) if e.color == Color.RED],
            Color.GREEN: [i for i, e in enumerate(emblems) if e.color == Color.GREEN],
            Color.BLUE: [i for i, e in enumerate(emblems) if e.color == Color.BLUE],
        }

        # 1. Green operations
        if op == "reroll_stat_green_all":
            for idx in color_indices[Color.GREEN]:
                emblems[idx].stat = self._sample_random_stat(role, Color.GREEN)
        elif op == "reroll_quality_green_all":
            for idx in color_indices[Color.GREEN]:
                emblems[idx].tier = self._sample_random_tier()
        elif op == "reroll_trait_green_all":
            for idx in color_indices[Color.GREEN]:
                emblems[idx].trait = self._sample_random_trait()
        elif op == "reroll_stat_green_first" and color_indices[Color.GREEN]:
            first_idx = color_indices[Color.GREEN][0]
            emblems[first_idx].stat = self._sample_random_stat(role, Color.GREEN)
        elif op == "reroll_stat_green_last" and color_indices[Color.GREEN]:
            last_idx = color_indices[Color.GREEN][-1]
            emblems[last_idx].stat = self._sample_random_stat(role, Color.GREEN)
        elif op == "reroll_stat_green_random" and color_indices[Color.GREEN]:
            rnd_idx = int(self.np_random.choice(color_indices[Color.GREEN]))
            emblems[rnd_idx].stat = self._sample_random_stat(role, Color.GREEN)

        # 2. Red operations
        elif op == "reroll_stat_red_all":
            for idx in color_indices[Color.RED]:
                emblems[idx].stat = self._sample_random_stat(role, Color.RED)
        elif op == "reroll_quality_red_all":
            for idx in color_indices[Color.RED]:
                emblems[idx].tier = self._sample_random_tier()
        elif op == "reroll_trait_red_all":
            for idx in color_indices[Color.RED]:
                emblems[idx].trait = self._sample_random_trait()
        elif op == "reroll_quality_red_first" and color_indices[Color.RED]:
            first_idx = color_indices[Color.RED][0]
            emblems[first_idx].tier = self._sample_random_tier()
        elif op == "reroll_quality_red_last" and color_indices[Color.RED]:
            last_idx = color_indices[Color.RED][-1]
            emblems[last_idx].tier = self._sample_random_tier()
        elif op == "reroll_quality_red_random" and color_indices[Color.RED]:
            rnd_idx = int(self.np_random.choice(color_indices[Color.RED]))
            emblems[rnd_idx].tier = self._sample_random_tier()

        # 3. Blue operations
        elif op == "reroll_stat_blue_all":
            for idx in color_indices[Color.BLUE]:
                emblems[idx].stat = self._sample_random_stat(role, Color.BLUE)
        elif op == "reroll_quality_blue_all":
            for idx in color_indices[Color.BLUE]:
                emblems[idx].tier = self._sample_random_tier()
        elif op == "reroll_trait_blue_all":
            for idx in color_indices[Color.BLUE]:
                emblems[idx].trait = self._sample_random_trait()
        elif op == "reroll_trait_blue_first" and color_indices[Color.BLUE]:
            first_idx = color_indices[Color.BLUE][0]
            emblems[first_idx].trait = self._sample_random_trait()
        elif op == "reroll_trait_blue_last" and color_indices[Color.BLUE]:
            last_idx = color_indices[Color.BLUE][-1]
            emblems[last_idx].trait = self._sample_random_trait()
        elif op == "reroll_trait_blue_random" and color_indices[Color.BLUE]:
            rnd_idx = int(self.np_random.choice(color_indices[Color.BLUE]))
            emblems[rnd_idx].trait = self._sample_random_trait()

        # 4. Other wildcard operations
        elif op == "increase_one_quality_random":
            upgradeable = [i for i, e in enumerate(emblems) if e.tier < 5]
            if upgradeable:
                target = int(self.np_random.choice(upgradeable))
                emblems[target].tier = min(5, emblems[target].tier + 1)

        elif op == "increase_two_reduce_one_quality_random":
            n = len(emblems)
            if n >= 2:
                # Randomly pick two emblems to increase
                inc_indices = self.np_random.choice(n, size=2, replace=False)
                for idx in inc_indices:
                    emblems[idx].tier = min(5, emblems[idx].tier + 1)
                # Randomly pick one emblem to reduce
                dec_idx = int(self.np_random.choice(n))
                emblems[dec_idx].tier = max(1, emblems[dec_idx].tier - 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        banner_idx = action // 3
        offer_idx = action % 3

        role = self.roles[banner_idx]
        op = self.current_offers[offer_idx]

        # Deduct token
        self.tokens -= 1

        # Check if action was valid according to mask
        mask = self.action_masks()
        was_valid = mask[action]

        if was_valid:
            self._apply_operation(role, op)

        new_score = calculate_total_fantasy_score(self.roster)
        # Reward is the change in score (scaled down to reasonable RL gradient scale)
        delta_score = new_score - self.current_score
        # Minor penalty for invalid actions to encourage following mask
        penalty = 0.0 if was_valid else -50.0
        reward = (delta_score / 100.0) + penalty
        self.current_score = new_score

        # Sample 3 new offers for the next turn
        self.current_offers = self._sample_offers()

        terminated = (self.tokens <= 0)
        truncated = False

        obs = self._get_obs()
        info = {
            "tokens": self.tokens,
            "score": self.current_score,
            "delta_score": delta_score,
            "action_mask": self.action_masks(),
            "applied_op": op,
            "role": role.value,
            "was_valid": was_valid,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Converts the current state into a normalized 1D numpy array."""
        features = []

        for role in self.roles:
            emblems = self.roster[role]
            for emblem in emblems:
                # 1. Stat index normalized
                available_stats = list(STAT_BASELINES[role][emblem.color].keys())
                stat_idx = available_stats.index(emblem.stat) if emblem.stat in available_stats else 0
                features.append(stat_idx / 5.0)

                # 2. Tier normalized (1..5 -> 0.2..1.0)
                features.append(emblem.tier / 5.0)

                # 3. Trait index normalized (0..4 -> 0.0..1.0)
                trait_idx = ALL_TRAITS.index(emblem.trait) if emblem.trait in ALL_TRAITS else 0
                features.append(trait_idx / 4.0)

        # Remaining tokens normalized
        features.append(self.tokens / float(self.initial_tokens))

        # 3 offered operations encoded as normalized IDs
        for op in self.current_offers:
            features.append(OP_TO_ID.get(op, 0) / float(len(OPERATIONS) - 1))

        return np.array(features, dtype=np.float32)

    def render(self):
        """Prints a human-readable summary of the current fantasy roster."""
        print(f"\n{'='*50}")
        print(f"Tokens remaining: {self.tokens}/{self.initial_tokens} | Total Score: {self.current_score:.1f}")
        print(f"Current offers: {self.current_offers}")
        print(f"{'='*50}")
        for role in self.roles:
            banner_score = calculate_banner_score(role, self.roster[role])
            print(f"[{role.value}] Score: {banner_score:.1f}")
            for i, e in enumerate(self.roster[role]):
                mult = TIER_MULTIPLIERS[e.tier]
                print(f"  Slot {i+1} ({e.color.value:5s}): Tier {e.tier} (x{mult:.2f}) | {e.stat:14s} | {e.trait.value}")
