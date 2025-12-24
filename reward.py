import numpy as np

"""Fast-clear reward shaping utilities for Super Mario Bros.

Design goals (aggressive shaping for fast clearing):
- Make forward progress (dx>0) the dominant signal.
- Strongly punish stagnation (dx==0) to break pipe-stuck behavior.
- Reward *forward jump* (dy>0 AND dx>0), not pogo-jumping in place.
- Heavily punish damage / death to prevent face-tanking enemies.
- Keep coins/score as mild auxiliary signals.
- Always be robust to missing keys in `info` / `prev_info`.
"""

def _get(info, key, default=0):
    try:
        return info.get(key, default)
    except Exception:
        return default

def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))

def _status_rank(status):
    # higher is better; robust to variations across implementations
    if status in ("fireball",):
        return 2
    if status in ("tall", "big"):
        return 1
    return 0

def forward_progress_reward(info, reward, prev_info,
                            forward_weight: float = 0.25,
                            backtrack_penalty: float = 0.35,
                            dx_clip: float = 5.0):
    """Strongly reward moving right; penalize moving left."""
    x = float(_get(info, "x_pos", 0))
    px = float(_get(prev_info, "x_pos", x))
    dx = _clip(x - px, -dx_clip, dx_clip)

    r = reward
    if dx > 0:
        r += forward_weight * dx
    elif dx < 0:
        r -= backtrack_penalty * abs(dx)
    return r, dx  # return dx for later use (e.g., forward-jump gating)

def stagnation_penalty_reward(dx, reward,
                              stagnation_penalty: float = 0.08):
    """Harsh penalty when not making horizontal progress (dx==0)."""
    r = reward
    if dx == 0:
        r -= stagnation_penalty
    return r

def forward_jump_reward(info, reward, prev_info, dx,
                        jump_bonus: float = 0.03,
                        fall_penalty: float = 0.01,
                        dy_clip: float = 6.0):
    """Reward upward motion only if moving forward; penalize falling when stuck."""
    y = float(_get(info, "y_pos", 0))
    py = float(_get(prev_info, "y_pos", y))
    dy = _clip(y - py, -dy_clip, dy_clip)

    r = reward
    if dy > 0 and dx > 0:
        r += jump_bonus * dy
    elif dy < 0 and dx <= 0:
        r -= fall_penalty * abs(dy)
    return r

def coin_reward(info, reward, prev_info, coin_weight: float = 2.0):
    """Small bonus for collecting coins (aux signal)."""
    coins = float(_get(info, "coins", 0))
    pcoins = float(_get(prev_info, "coins", coins))
    dcoins = coins - pcoins
    return reward + coin_weight * dcoins

def score_reward(info, reward, prev_info, score_weight: float = 0.002):
    """Small bonus for increasing in-game score (often enemies/items)."""
    score = float(_get(info, "score", 0))
    pscore = float(_get(prev_info, "score", score))
    dscore = score - pscore
    if dscore > 0:
        return reward + score_weight * dscore
    return reward

def time_penalty_reward(reward, per_step_penalty: float = 0.02):
    """Per-step penalty to encourage faster completion."""
    return reward - per_step_penalty

def damage_death_penalty_reward(info, reward, prev_info,
                                life_drop_penalty: float = 50.0,
                                status_drop_penalty: float = 25.0):
    """Heavily punish getting hurt or losing a life."""
    r = reward

    life = _get(info, "life", None)
    plife = _get(prev_info, "life", life)
    try:
        if (life is not None) and (plife is not None) and (life < plife):
            r -= life_drop_penalty
    except Exception:
        pass

    status = _get(info, "status", None)
    pstatus = _get(prev_info, "status", status)
    try:
        if (status is not None) and (pstatus is not None):
            if _status_rank(status) < _status_rank(pstatus):
                r -= status_drop_penalty
    except Exception:
        pass

    return r

def final_flag_reward(info, reward, flag_bonus: float = 500.0):
    """Big bonus when reaching the flag."""
    flag_get = bool(_get(info, "flag_get", False))
    return reward + (flag_bonus if flag_get else 0.0)

def shaped_reward(info, env_reward, prev_info):
    """Apply all shaping in a single call (aggressive, fast-clear)."""
    r = float(env_reward)

    # 1) Progress dominates
    r, dx = forward_progress_reward(info, r, prev_info)

    # 2) Break pipe-stuck behavior
    r = stagnation_penalty_reward(dx, r)

    # 3) Encourage forward jump, not pogo jumping
    r = forward_jump_reward(info, r, prev_info, dx)

    # 4) Mild aux signals
    r = coin_reward(info, r, prev_info)
    r = score_reward(info, r, prev_info)

    # 5) Finish faster
    r = time_penalty_reward(r)

    # 6) Avoid taking damage / dying
    r = damage_death_penalty_reward(info, r, prev_info)

    # 7) Clear
    r = final_flag_reward(info, r)

    return r
