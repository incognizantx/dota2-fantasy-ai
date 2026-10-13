"""
Main entry point: Simulation and Baseline Agent demonstration for Dota 2 Fantasy League.
"""

import numpy as np
from fantasy_env import DotaFantasyEnv
from scoring import calculate_total_fantasy_score


def run_greedy_baseline_agent(env: DotaFantasyEnv, seed: int = 42):
    """
    A 1-step Lookahead (Greedy) baseline agent.
    At each step, evaluates all valid actions from the action mask,
    simulates the expected immediate gain, and picks the best one.
    """
    obs, info = env.reset(seed=seed)
    print("=" * 60)
    print("INITIAL ROSTER (Before spending 40 tokens):")
    env.render()

    initial_score = info["score"]
    step_num = 0

    while True:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]

        best_action = valid_actions[0]
        best_delta = -float("inf")

        # Evaluate candidate actions greedily
        current_state_score = env.current_score
        for act in valid_actions:
            banner_idx = act // 3
            offer_idx = act % 3
            role = env.roles[banner_idx]
            op = env.current_offers[offer_idx]

            # Clone roster to simulate effect
            temp_roster = {
                r: [e.copy() for e in env.roster[r]]
                for r in env.roles
            }

            # Temporary env simulation
            sim_env = DotaFantasyEnv(stage=env.stage, initial_tokens=env.tokens)
            sim_env.roster = temp_roster
            sim_env._apply_operation(role, op)
            sim_score = calculate_total_fantasy_score(sim_env.roster)
            delta = sim_score - current_state_score

            if delta > best_delta:
                best_delta = delta
                best_action = act

        obs, reward, terminated, truncated, info = env.step(best_action)
        step_num += 1

        if step_num % 10 == 0 or terminated:
            print(f"\n[Step {step_num}/40] Applied: {info['applied_op']} on {info['role']} | Score: {info['score']:.1f} (Delta: {info['delta_score']:+.1f})")

        if terminated:
            break

    print("\n" + "=" * 60)
    print("FINAL ROSTER (After spending 40 tokens):")
    env.render()
    print("=" * 60)
    total_gain = env.current_score - initial_score
    print(f"Initial Score: {initial_score:.1f}")
    print(f"Final Score:   {env.current_score:.1f}")
    print(f"Total Gain:    +{total_gain:.1f} (+{(total_gain / initial_score)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    env = DotaFantasyEnv(stage=1, initial_tokens=40)
    run_greedy_baseline_agent(env, seed=42)
