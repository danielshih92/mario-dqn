import os
import torch
# 舊的 gym (給 Mario 用)
import gym as old_gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# 新的 gymnasium (給 SB3 用)
import gymnasium as new_gym
# 【修正】這裡改成 GrayscaleObservation (小寫 s)
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
import shimmy

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ================= Config =================
ENV_ID = "SuperMarioBros-1-1-v0"
SAVE_DIR = "ckpt_ppo"
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 2.5e-4
N_ENVS = 4                  
SAVE_FREQ = 50_000

REDUCED_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]

# ================= Wrappers =================
class SkipFrame(new_gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        
        return obs, total_reward, terminated, truncated, info

def make_env(env_id, rank, seed=0):
    def _init():
        # 1. 建立原始環境 (Old Gym)
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, REDUCED_MOVEMENT)
        
        # 2. 透過 shimmy 轉換成 Gymnasium
        env = shimmy.GymV21CompatibilityV0(env=env)
        
        # 3. 使用 Gymnasium Wrappers
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, (84, 84))
        
        env = Monitor(env)
        
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[INFO] Launching {N_ENVS} parallel environments with Shimmy compatibility...")
    env = SubprocVecEnv([make_env(ENV_ID, i) for i in range(N_ENVS)])
    
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/",
        device="cuda"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),
        save_path=SAVE_DIR,
        name_prefix="mario_ppo"
    )

    print("[INFO] Start training...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    
    model.save(os.path.join(SAVE_DIR, "mario_ppo_final"))
    print(f"[INFO] Model saved to {SAVE_DIR}")
    env.close()

if __name__ == "__main__":
    main()