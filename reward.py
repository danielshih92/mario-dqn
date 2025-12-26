import numpy as np

"""
Reward shaping for fast clear (1-1).

Core idea:
- Reward forward progress strongly (x_pos delta)
- Penalize stagnation (stuck)
- Small time penalty
- Big reward when flag_get
- Penalize death / damage (if keys exist)
- Optional nonlinear reward transform (stabilizes learning)
"""

def _get(info, key, default=0):
    try:
        return info.get(key, default)
    except Exception:
        return default


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def _status_rank(status):
    if status in ("fireball",):
        return 2
    if status in ("tall", "big"):
        return 1
    return 0


def reward_transform(r: float) -> float:
    # Common trick (stabilize large sparse rewards):
    # sign(r) * (sqrt(|r|+1)-1) + 0.001*r
    return float(np.sign(r) * (np.sqrt(abs(r) + 1.0) - 1.0) + 0.001 * r)


def shaped_reward(info, env_reward, prev_info):
    r = float(env_reward)

    x = float(_get(info, "x_pos", 0.0))
    px = float(_get(prev_info, "x_pos", x))
    dx = _clip(x - px, -5.0, 5.0)

    # 1) progress dominates
    if dx > 0:
        r += 0.25 * dx
    elif dx < 0:
        r -= 0.35 * abs(dx)

    # 2) stagnation penalty
    if dx == 0:
        r -= 0.08

    # 3) forward jump bonus (only if moving forward)
    y = float(_get(info, "y_pos", 0.0))
    py = float(_get(prev_info, "y_pos", y))
    dy = _clip(y - py, -6.0, 6.0)
    if dy > 0 and dx > 0:
        r += 0.03 * dy

    # 4) mild aux signals
    coins = float(_get(info, "coins", 0.0))
    pcoins = float(_get(prev_info, "coins", coins))
    if coins > pcoins:
        r += 2.0 * (coins - pcoins)

    score = float(_get(info, "score", 0.0))
    pscore = float(_get(prev_info, "score", score))
    dscore = score - pscore
    if dscore > 0:
        r += 0.002 * dscore

    # 5) per-step time penalty
    r -= 0.02

    # 6) damage / death penalty (if available)
    life = _get(info, "life", None)
    plife = _get(prev_info, "life", life)
    try:
        if (life is not None) and (plife is not None) and (life < plife):
            r -= 50.0
    except Exception:
        pass

    status = _get(info, "status", None)
    pstatus = _get(prev_info, "status", status)
    try:
        if (status is not None) and (pstatus is not None):
            if _status_rank(status) < _status_rank(pstatus):
                r -= 25.0
    except Exception:
        pass

    # 7) clear bonus
    if bool(_get(info, "flag_get", False)):
        r += 500.0

    # final stabilization transform
    r = reward_transform(r)
    return r

