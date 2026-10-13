"""
Unit tests for Dota 2 Fantasy Environment and Scoring.
"""

import pytest
import numpy as np
from fantasy_data import Color, Role, Trait, STAGE_1_LAYOUT
from scoring import (
    Emblem,
    calculate_banner_trait_bonuses,
    calculate_banner_score,
    calculate_total_fantasy_score,
)
from fantasy_env import DotaFantasyEnv


def test_trait_scoring():
    # Test Friendly bonus
    emblems = [
        Emblem(Color.RED, "Creeps", 1, Trait.FRIENDLY),
        Emblem(Color.GREEN, "Teamfight", 1, Trait.FRIENDLY),
        Emblem(Color.RED, "GPM", 1, Trait.UNIQUE),
    ]
    bonuses = calculate_banner_trait_bonuses(emblems)
    # 2 Friendly traits -> both should get +0.50
    assert bonuses[0] == pytest.approx(0.50)
    assert bonuses[1] == pytest.approx(0.50)
    # Unique trait does NOT trigger because there are 2 Friendly traits (not all unique)
    assert bonuses[2] == pytest.approx(0.0)


def test_vampiric_trait_scoring():
    # Test Vampiric bonus: +50% self, -25% adjacent
    emblems = [
        Emblem(Color.RED, "Creeps", 1, Trait.VAMPIRIC),
        Emblem(Color.GREEN, "Teamfight", 1, Trait.BENEVOLENT),
        Emblem(Color.RED, "GPM", 1, Trait.FRACTAL),
    ]
    bonuses = calculate_banner_trait_bonuses(emblems)
    # Index 0 (Vampiric): +0.50 self + 0.20 from Benevolent neighbor = +0.70
    assert bonuses[0] == pytest.approx(0.70)
    # Index 1 (Benevolent): -0.25 from Vampiric neighbor = -0.25
    assert bonuses[1] == pytest.approx(-0.25)
    # Index 2 (Fractal): all tiers are Tier 1 (not unique), +0.20 from Benevolent neighbor = +0.20
    assert bonuses[2] == pytest.approx(0.20)


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
