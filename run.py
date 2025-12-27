import os
import time
import glob
import numpy as np
import torch
from tqdm import tqdm

from utils import make_mario_env
from reward import shaped_reward
from model import DuelingCNN
from DQN import DQNAgent, ReplayMemory


# ===================== Config =====================
ENV_ID = "SuperMarioBros-1-1-v0"

STACK_K = 4
FRAME_SKIP = 4
REDUCED_ACTIONS = True

LR = 1e-4
GAMMA = 0.99

MEMORY_SIZE = 100_000
BATCH_SIZE = 256
WARMUP_STEPS = 10_000          # collect experience before learning

TARGET_UPDATE_STEPS = 5_000    # sync target network every N gradient steps
GRAD_CLIP = 10.0

TOTAL_EPISODES = 2000
MAX_STEPS_PER_EP = 5000        # safety cap

# epsilon schedule (fast decay)
EPS_START = 0.05 #1.0
EPS_END = 0.01 #0.05
EPS_DECAY_EPISODES = 200       # decay quickly to reduce training time

# early stop if stuck
MAX_STAGNATION_STEPS = 400

# evaluation
EVAL_INTERVAL = 50
MAX_EVAL_STEPS = 5000

# checkpoints (Google Drive path)
SAVE_DIR = "/content/drive/MyDrive/mario/ckpt_r2"
SAVE_EVERY = 200               # periodic checkpoint
BEST_NAME = "best.pth"

RESUME = True
RESUME_PATH = "/content/drive/MyDrive/mario/ckpt/best_400.pth"


def epsilon_by_episode(ep: int) -> float:
    if ep <= 0:
        return EPS_START
    if ep >= EPS_DECAY_EPISODES:
        return EPS_END
    ratio = ep / float(EPS_DECAY_EPISODES)
    return EPS_START + ratio * (EPS_END - EPS_START)


@torch.no_grad()
def run_eval(agent: DQNAgent, episode_idx: int):
    env = make_mario_env(ENV_ID, stack_k=STACK_K, skip=FRAME_SKIP, reduced_actions=REDUCED_ACTIONS)
    obs = env.reset()
    done = False
    steps = 0

    prev_info = {"x_pos": 0, "y_pos": 0, "score": 0, "coins": 0, "time": 400, "flag_get": False, "life": 3}
    total_env_reward = 0.0
    total_shaped_reward = 0.0

    while (not done) and (steps < MAX_EVAL_STEPS):
        agent.epsilon = 0.0
        action = agent.act(obs, deterministic=True)
        next_obs, env_r, done, info = env.step(action)

        r = shaped_reward(info, env_r, prev_info)
        prev_info = dict(info)

        total_env_reward += float(env_r)
        total_shaped_reward += float(r)

        obs = next_obs
        steps += 1

    env.close()
    return {
        "steps": steps,
        "env_reward": total_env_reward,
        "shaped_reward": total_shaped_reward,
        "flag_get": bool(prev_info.get("flag_get", False)),
        "x_pos": int(prev_info.get("x_pos", 0)),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # make train env
    env = make_mario_env(ENV_ID, stack_k=STACK_K, skip=FRAME_SKIP, reduced_actions=REDUCED_ACTIONS)

    obs_shape = (STACK_K, 84, 84)
    n_actions = env.action_space.n
    print("[Env] actions =", n_actions, "obs_shape =", obs_shape)

    agent = DQNAgent(
        model_cls=DuelingCNN,
        state_dim=obs_shape,
        action_dim=n_actions,
        lr=LR,
        gamma=GAMMA,
        target_update_steps=TARGET_UPDATE_STEPS,
        device=device,
        grad_clip=GRAD_CLIP,
    )

    if RESUME and os.path.exists(RESUME_PATH):
        print(f"[RESUME] loading from {RESUME_PATH}")
        agent.load(RESUME_PATH)
        agent.epsilon = 0.2

    memory = ReplayMemory(MEMORY_SIZE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_score = -1e18
    global_env_steps = 0
    global_updates = 0

    # warmup prev_info (per episode)
    for ep in tqdm(range(1, TOTAL_EPISODES + 1), desc="Training"):
        agent.epsilon = epsilon_by_episode(ep)

        obs = env.reset()
        done = False
        steps = 0

        prev_info = {"x_pos": 0, "y_pos": 0, "score": 0, "coins": 0, "time": 400, "flag_get": False, "life": 3}
        stagnation = 0

        ep_env_reward = 0.0
        ep_shaped_reward = 0.0
        ep_losses = []

        while (not done) and (steps < MAX_STEPS_PER_EP):
            action = agent.act(obs, deterministic=False)
            next_obs, env_r, done, info = env.step(action)

            r = shaped_reward(info, env_r, prev_info)
            ep_env_reward += float(env_r)
            ep_shaped_reward += float(r)

            # stagnation early stop
            if info.get("x_pos", 0) == prev_info.get("x_pos", 0):
                stagnation += 1
                if stagnation >= MAX_STAGNATION_STEPS:
                    done = True
            else:
                stagnation = 0

            prev_info = dict(info)

            memory.push(obs, action, r, next_obs, done)
            obs = next_obs

            global_env_steps += 1
            steps += 1

            # learn after warmup and when enough samples
            if (global_env_steps > WARMUP_STEPS) and (len(memory) >= BATCH_SIZE):
                batch = memory.sample(BATCH_SIZE)
                loss = agent.learn(batch)
                ep_losses.append(loss)
                global_updates += 1

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # periodic save
        if ep % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"ep_{ep}.pth")
            agent.save(ckpt_path)

        # evaluation
        if ep % EVAL_INTERVAL == 0:
            stats = run_eval(agent, ep)
            print(
                f"[EVAL ep={ep}] steps={stats['steps']} "
                f"x={stats['x_pos']} flag={stats['flag_get']} "
                f"envR={stats['env_reward']:.1f} shapedR={stats['shaped_reward']:.1f}"
            )

            # keep best by env reward (you can also use x_pos / flag_get)
            score = stats["env_reward"]
            if score > best_score:
                best_score = score

                # remove old best_*.pth
                for old in glob.glob(os.path.join(SAVE_DIR, "best_*.pth")):
                    try:
                        os.remove(old)
                    except OSError:
                        pass

                best_tag = os.path.join(SAVE_DIR, f"best_{ep}.pth")
                agent.save(best_tag)
                agent.save(os.path.join(SAVE_DIR, BEST_NAME))
                print(f"[BEST] saved: {best_tag}")

        print(
            f"[TRAIN ep={ep}] eps={agent.epsilon:.3f} "
            f"envR={ep_env_reward:.1f} shapedR={ep_shaped_reward:.1f} "
            f"avgLoss={avg_loss:.4f} mem={len(memory)}"
        )

    env.close()


if __name__ == "__main__":
    main()