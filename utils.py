import numpy as np
import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


# =====================
# Common action sets
# =====================
REDUCED_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]


def _unwrap_obs_reset(out):
    # gymnasium reset() returns (obs, info); gym returns obs
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _unwrap_obs_step(out):
    # gymnasium step() returns 5-tuple; gym returns 4-tuple
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    raise RuntimeError(f"Unexpected step() output: {type(out)} {out}")


def preprocess_frame(frame, out_size=(84, 84)):
    """RGB -> Gray -> Resize -> float32 [0,1]. Output shape: (84, 84)"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame


class FrameStackEnv(gym.Wrapper):
    """
    - Action repeat (frame-skip): repeat same action N frames
    - Max pooling over last two frames (helps with flicker)
    - Frame stacking: stack last K processed grayscale frames
    Output obs shape: (K, 84, 84) float32
    """
    def __init__(self, env, stack_k=4, skip=4):
        super().__init__(env)
        self.stack_k = int(stack_k)
        self.skip = int(skip)
        self.frames = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = _unwrap_obs_reset(out)
        proc = preprocess_frame(obs)
        self.frames = [proc for _ in range(self.stack_k)]
        return np.stack(self.frames, axis=0)  # (K,84,84)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        last_obs = None
        second_last_obs = None

        for _ in range(self.skip):
            out = self.env.step(action)
            obs, r, done, info = _unwrap_obs_step(out)
            total_reward += float(r)

            second_last_obs = last_obs
            last_obs = obs

            if done:
                break

        # Max-pool over last two raw frames (if available)
        if second_last_obs is None:
            pooled = last_obs
        else:
            pooled = np.maximum(second_last_obs, last_obs)

        proc = preprocess_frame(pooled)

        # update stack
        self.frames.pop(0)
        self.frames.append(proc)

        stacked = np.stack(self.frames, axis=0)  # (K,84,84)
        return stacked, total_reward, done, info


def make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    stack_k=4,
    skip=4,
    reduced_actions=True,
):
    """
    Returns an env with:
      - TimeLimit unwrapped (prevents 5-tuple expectation crash)
      - JoypadSpace (discrete action)
      - FrameSkip + MaxPool + FrameStack
      - Obs: (K,84,84) float32
    """
    env = gym_super_mario_bros.make(env_id)

    # IMPORTANT:
    # Gym 0.26 TimeLimit expects new step API (5-tuple),
    # but nes-py / mario env returns old 4-tuple => crash.
    # So unwrap TimeLimit if present.
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    movement = REDUCED_MOVEMENT if reduced_actions else None
    if movement is None:
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        movement = SIMPLE_MOVEMENT

    env = JoypadSpace(env, movement)
    env = FrameStackEnv(env, stack_k=stack_k, skip=skip)
    return env
