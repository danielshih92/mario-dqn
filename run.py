import os
import time
import numpy as np
import torch
import cv2
from tqdm import tqdm
import glob

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import StepAPICompatibility
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from reward import shaped_reward
from model import CustomCNN
from DQN import DQN, ReplayMemory


# ===================== Config =====================
ENV_ID = "SuperMarioBros-1-1-v0"

LR = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
MEMORY_SIZE = 50_000
TARGET_UPDATE = 2_000          # in gradient steps
TOTAL_EPISODES = 1200 #2000

# Exploration schedule
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_EPISODES = 1200      # linearly decay over first N episodes

# Rendering / evaluation
TRAIN_RENDER = False           # training render is slow; keep False
EVAL_INTERVAL = 50 #500            # run an eval episode every N episodes
MAX_EVAL_STEPS = 5000

# Early stop if stuck
MAX_STAGNATION_STEPS = 500

SAVE_DIR = "/content/drive/MyDrive/mario/ckpt"
SAVE_EVERY = 200               # save checkpoint every N episodes (in addition to best)

EP_OFFSET = 2700 # to keep track of actual episode number when resuming training
RESUME = True
RESUME_PATH = "/content/drive/MyDrive/mario/ckpt/best_2700.pth"

# Reduce action space (often helps early learning)
USE_REDUCED_ACTIONS = True
REDUCED_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]


def make_env():
    env = gym_super_mario_bros.make(ENV_ID)

    # Unwrap TimeLimit if present (StepAPICompatibility expects older API)
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    env = StepAPICompatibility(env, new_step_api=False)

    movement = REDUCED_MOVEMENT if USE_REDUCED_ACTIONS else SIMPLE_MOVEMENT
    env = JoypadSpace(env, movement)
    return env


def epsilon_by_episode(ep: int) -> float:
    if ep <= 0:
        return EPS_START
    if ep >= EPS_DECAY_EPISODES:
        return EPS_END
    ratio = ep / float(EPS_DECAY_EPISODES)
    return EPS_START + ratio * (EPS_END - EPS_START)


@torch.no_grad()
def run_eval_episode(dqn: DQN, episode_idx: int):
    env = make_env()
    frames = []
    state = env.reset()
    done = False
    prev_info = {"x_pos": 0, "y_pos": 0, "score": 0, "coins": 0, "time": 400, "flag_get": False, "life": 3}
    stagnation = 0

    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)

    total_reward = 0.0
    total_env_reward = 0.0
    steps = 0

    while (not done) and (steps < MAX_EVAL_STEPS):
        # deterministic greedy action
        action = dqn.take_action(state, deterministic=True)
        next_state, env_reward, done, info = env.step(action)

        if info.get("x_pos", 0) == prev_info.get("x_pos", 0):
            stagnation += 1
        else:
            stagnation = 0

        r = shaped_reward(info, env_reward, prev_info, stagnation)
        total_reward += float(r)
        total_env_reward += float(env_reward)

        prev_info = dict(info)

        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)
        state = next_state
        steps += 1

    env.close()

    return {
        "steps": steps,
        "total_shaped_reward": total_reward,
        "total_env_reward": total_env_reward,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()

    obs_shape = (1, 84, 84)
    n_actions = env.action_space.n

    dqn = DQN(
        model=CustomCNN,
        state_dim=obs_shape,
        action_dim=n_actions,
        learning_rate=LR,
        gamma=GAMMA,
        epsilon=EPS_START,
        target_update=TARGET_UPDATE,
        device=device,
    )
    if RESUME and os.path.exists(RESUME_PATH):
        print(f"[RESUME] loading model from {RESUME_PATH}")
        dqn.q_net.load_state_dict(torch.load(RESUME_PATH))
        dqn.epsilon = 0.3

    memory = ReplayMemory(MEMORY_SIZE)

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_eval_score = -1e18
    global_step = 0

    for ep in tqdm(range(1, TOTAL_EPISODES + 1), desc="Training"):
        effective_ep = EP_OFFSET + ep
        state = env.reset()
        state = preprocess_frame(state)
        state = np.expand_dims(state, axis=0)

        done = False
        prev_info = {"x_pos": 0, "y_pos": 0, "score": 0, "coins": 0, "time": 400, "flag_get": False, "life": 3}
        stagnation = 0

        dqn.epsilon = epsilon_by_episode(effective_ep)

        ep_env_reward = 0.0
        ep_shaped_reward = 0.0

        while not done:
            action = dqn.take_action(state, deterministic=False)
            next_state, env_reward, done, info = env.step(action)

            # update stagnation first (used for cumulative penalty and early stop)
            if info.get("x_pos", 0) == prev_info.get("x_pos", 0):
                stagnation += 1
                if stagnation >= MAX_STAGNATION_STEPS:
                    done = True
            else:
                stagnation = 0

            # shaped reward (with cumulative stagnation penalty)
            r = shaped_reward(info, env_reward, prev_info, stagnation)

            ep_env_reward += float(env_reward)
            ep_shaped_reward += float(r)

            prev_info = dict(info)

            next_state_proc = preprocess_frame(next_state)
            next_state_proc = np.expand_dims(next_state_proc, axis=0)

            memory.push(state, action, r, next_state_proc, done)
            state = next_state_proc

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                state_dict = {
                    "states": batch[0],
                    "actions": batch[1],
                    "rewards": batch[2],
                    "next_states": batch[3],
                    "dones": batch[4],
                }
                dqn.train_per_step(state_dict)
                global_step += 1

            if TRAIN_RENDER:
                env.render()

        # periodic save
        if effective_ep % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"ep_{effective_ep}.pth")
            torch.save(dqn.q_net.state_dict(), ckpt_path)

        # periodic evaluation (with optional video)
        if effective_ep % EVAL_INTERVAL == 0:
            eval_stats = run_eval_episode(dqn, effective_ep)
            print(
                f"[EVAL ep={effective_ep}] steps={eval_stats['steps']} "
                f"env_reward={eval_stats['total_env_reward']:.1f} "
                f"shaped_reward={eval_stats['total_shaped_reward']:.1f} "
            )

            # keep best
            if eval_stats["total_env_reward"] > best_eval_score:
                best_eval_score = eval_stats["total_env_reward"]

                # 1) 先刪掉舊的 best_*.pth
                for old in glob.glob(os.path.join(SAVE_DIR, "best_*.pth")):
                    try:
                        os.remove(old)
                    except OSError:
                        pass

                # 2) 存新的 best_{episode}.pth（episode 用「實際總回合」命名，下面第3點會處理 offset）
                best_tag_path = os.path.join(SAVE_DIR, f"best_{effective_ep}.pth")
                torch.save(dqn.q_net.state_dict(), best_tag_path)

                # 3) 同步存一份固定檔名 best.pth，方便下次 RESUME
                best_path = os.path.join(SAVE_DIR, "best.pth")
                torch.save(dqn.q_net.state_dict(), best_path)

                print(f"[BEST] new best saved: {best_tag_path}")

        print(
            f"[TRAIN ep={ep}] eps={dqn.epsilon:.3f} "
            f"env_reward={ep_env_reward:.1f} shaped_reward={ep_shaped_reward:.1f}"
        )

    env.close()


if __name__ == "__main__":
    main()
