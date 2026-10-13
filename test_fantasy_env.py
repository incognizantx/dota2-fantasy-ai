"""
Unit tests for Dota 2 Fantasy Environment and Scoring.
Matches verified formulas against in-game screenshots.
"""

import pytest
import numpy as np
from fantasy_data import Color, Role, Trait, TIER_BONUSES
from scoring import (
    Emblem,
    calculate_banner_trait_bonuses,
    calculate_emblem_multiplier,
    calculate_banner_score,
    calculate_total_fantasy_score,
)
from fantasy_env import DotaFantasyEnv


def test_screenshot_exact_matching():
    """
    Validates calculations against the exact values shown on in-game screenshot:
    Core:
      Slot 1 (RED): DEATHS Tier V (150%) + Vampiric (50%) = 300%
      Slot 2 (GREEN): TEAMFIGHT Tier I (10%) + Incorruptible (60%) = 170%
      Slot 3 (RED): GPM Tier V (150%) + Benevolent (0%) = 250%
    Mid:
      Slot 1 (RED): GPM Tier V (150%) + Unique (50%) = 300%
      Slot 2 (BLUE): OBS WARDS Tier III (60%) + Benevolent (20%) = 180%
      Slot 3 (GREEN): STUNS Tier IV (100%) + Benevolent (20%) = 220%
    """
    # Core banner
    core_emblems = [
        Emblem(Color.RED, "Deaths", 5, Trait.VAMPIRIC),
        Emblem(Color.GREEN, "Teamfight", 1, Trait.INCORRUPTIBLE),
        Emblem(Color.RED, "GPM", 5, Trait.BENEVOLENT),
    ]
    core_bonuses = calculate_banner_trait_bonuses(core_emblems)
    assert core_bonuses[0] == pytest.approx(0.50)  # Vampiric +50%
    assert core_bonuses[1] == pytest.approx(0.60)  # Incorruptible on T1: +60%
    assert core_bonuses[2] == pytest.approx(0.00)  # Benevolent standalone: 0%

    core_mults = [calculate_emblem_multiplier(e, b) for e, b in zip(core_emblems, core_bonuses)]
    assert core_mults[0] == pytest.approx(3.00)  # 300%
    assert core_mults[1] == pytest.approx(1.70)  # 170%
    assert core_mults[2] == pytest.approx(2.50)  # 250%

    # Mid banner
    mid_emblems = [
        Emblem(Color.RED, "GPM", 5, Trait.UNIQUE),
        Emblem(Color.BLUE, "Wards", 3, Trait.BENEVOLENT),
        Emblem(Color.GREEN, "Stuns", 4, Trait.BENEVOLENT),
    ]
    mid_bonuses = calculate_banner_trait_bonuses(mid_emblems)
    assert mid_bonuses[0] == pytest.approx(0.50)  # Unique standalone on banner: +50%
    assert mid_bonuses[1] == pytest.approx(0.20)  # Benevolent adjacent: +20%
    assert mid_bonuses[2] == pytest.approx(0.20)  # Benevolent adjacent: +20%

    mid_mults = [calculate_emblem_multiplier(e, b) for e, b in zip(mid_emblems, mid_bonuses)]
    assert mid_mults[0] == pytest.approx(3.00)  # 300%
    assert mid_mults[1] == pytest.approx(1.80)  # 180%
    assert mid_mults[2] == pytest.approx(2.20)  # 220%


def test_friendly_trait_stacking():
    # 2 Friendly traits -> both should get +0.50 (+50%)
    emblems = [
        Emblem(Color.RED, "Creeps", 2, Trait.FRIENDLY),
        Emblem(Color.GREEN, "Teamfight", 2, Trait.FRIENDLY),
        Emblem(Color.RED, "GPM", 1, Trait.VAMPIRIC),
    ]
    bonuses = calculate_banner_trait_bonuses(emblems)
    assert bonuses[0] == pytest.approx(0.50)
    assert bonuses[1] == pytest.approx(0.50)
    assert bonuses[2] == pytest.approx(0.50)


def test_env_initialization_and_reset():
    env = DotaFantasyEnv(stage=1, initial_tokens=40)
    obs, info = env.reset(seed=42)

    assert obs.shape == env.observation_space.shape
    assert info["tokens"] == 40
    assert "score" in info
    assert "action_mask" in info
    assert len(info["action_mask"]) == 9


def test_action_masking_core_banner():
    env = DotaFantasyEnv(stage=1, initial_tokens=40)
    env.reset(seed=42)

    # Force offers to contain blue operation
    env.current_offers = [
        "reroll_stat_blue_all",
        "reroll_stat_red_all",
        "increase_one_quality_random",
    ]

    mask = env.action_masks()
    # Core is banner 0 (Red-Green-Red). Blue reroll is action 0 (0 * 3 + 0).
    # Since Core has no blue emblems, action 0 MUST be masked (False)
    assert mask[0] == False
    # Red reroll is action 1 (0 * 3 + 1). Core HAS red emblems -> True
    assert mask[1] == True


def test_full_episode_rollout():
    env = DotaFantasyEnv(stage=1, initial_tokens=40)
    obs, info = env.reset(seed=123)

    initial_score = info["score"]
    steps = 0
    terminated = False

    while not terminated:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        # Agent picks random valid action
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps == 40
    assert info["tokens"] == 0
    assert terminated == True
    assert env.tokens == 0
