import os
import numpy as np
import torch

from utils import make_mario_env
from model import DuelingCNN
from DQN import DQNAgent


# ========== Config ==========
ENV_ID = "SuperMarioBros-1-1-v0"
STACK_K = 4
FRAME_SKIP = 4
REDUCED_ACTIONS = True

MODEL_PATH = "best.pth"  # 你可以改成你的 ckpt 路徑，例如：/content/drive/MyDrive/mario/ckpt/best.pth
TOTAL_EPISODES = 5
VISUALIZE = True
MAX_STEPS = 6000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_mario_env(ENV_ID, stack_k=STACK_K, skip=FRAME_SKIP, reduced_actions=REDUCED_ACTIONS)

    obs_shape = (STACK_K, 84, 84)
    n_actions = env.action_space.n

    agent = DQNAgent(
        model_cls=DuelingCNN,
        state_dim=obs_shape,
        action_dim=n_actions,
        lr=1e-4,
        gamma=0.99,
        target_update_steps=5000,
        device=device,
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    for ep in range(1, TOTAL_EPISODES + 1):
        obs = env.reset()
        done = False
        total_env_reward = 0.0
        steps = 0

        while (not done) and (steps < MAX_STEPS):
            action = agent.act(obs, deterministic=True)
            obs, env_r, done, info = env.step(action)
            total_env_reward += float(env_r)
            steps += 1

            if VISUALIZE:
                env.render()

        print(f"[EVAL] ep={ep} steps={steps} env_reward={total_env_reward:.1f} x={info.get('x_pos', 0)} flag={info.get('flag_get', False)}")

    env.close()


if __name__ == "__main__":
    main()
