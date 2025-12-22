import numpy as np

"""Reward shaping utilities for Super Mario Bros.

These functions take the original environment reward and add additional shaping
signals based on the `info` dict returned by the environment.

Notes:
- Keep shaping magnitudes modest relative to the environment's own rewards.
- Always be robust to missing keys in `info` / `prev_info`.
"""
def _get(info, key, default=0):
    try:
        return info.get(key, default)
    except Exception:
        return default

def get_coin_reward(info, reward, prev_info, coin_weight: float = 10.0):
    """Bonus for collecting coins."""
    coins = _get(info, "coins", 0)
    prev_coins = _get(prev_info, "coins", 0)
    return reward + (coins - prev_coins) * coin_weight

def distance_x_offset_reward(info, reward, prev_info,
                             forward_weight: float = 0.1,
                             backtrack_penalty: float = 0.2,
                             stagnation_penalty: float = 0.01):
    """Encourage moving right; penalize moving left / stagnation."""
    x = _get(info, "x_pos", 0)
    prev_x = _get(prev_info, "x_pos", 0)
    dx = x - prev_x

    shaped = reward
    if dx > 0:
        shaped += forward_weight * dx
    elif dx < 0:
        shaped -= backtrack_penalty * abs(dx)
    else:
        shaped -= stagnation_penalty
    return shaped

def distance_y_offset_reward(info, reward, prev_info,
                             jump_bonus: float = 0.02,
                             fall_penalty: float = 0.005,
                             clip: float = 5.0):
    """Small incentive for upward motion (jumping over obstacles)."""
    y = _get(info, "y_pos", 0)
    prev_y = _get(prev_info, "y_pos", 0)
    dy = y - prev_y
    dy = float(np.clip(dy, -clip, clip))

    shaped = reward
    if dy > 0:
        shaped += jump_bonus * dy
    elif dy < 0:
        shaped -= fall_penalty * abs(dy)
    return shaped

def monster_score_reward(info, reward, prev_info, score_weight: float = 0.01):
    """Encourage increasing the in-game score (often from enemies / items)."""
    score = _get(info, "score", 0)
    prev_score = _get(prev_info, "score", 0)
    dscore = score - prev_score
    return reward + score_weight * dscore

def time_penalty_reward(info, reward, prev_info,
                        per_step_penalty: float = 0.01):
    """Encourage faster completion by applying a small per-step penalty."""
    # The env's `time` counts down; we just apply a constant small penalty.
    return reward - per_step_penalty

def final_flag_reward(info, reward,
                      flag_bonus: float = 500.0):
    """Big bonus when reaching the flag."""
    flag_get = bool(_get(info, "flag_get", False))
    return reward + (flag_bonus if flag_get else 0.0)

def shaped_reward(info, env_reward, prev_info):
    """Convenience: apply all shaping in a single call."""
    r = env_reward
    r = get_coin_reward(info, r, prev_info)
    r = distance_x_offset_reward(info, r, prev_info)
    r = distance_y_offset_reward(info, r, prev_info)
    r = monster_score_reward(info, r, prev_info)
    r = time_penalty_reward(info, r, prev_info)
    r = final_flag_reward(info, r)
    return r
