import os
import gym
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# ================= Config =================
ENV_ID = "SuperMarioBros-1-1-v0"
SAVE_DIR = "ckpt_ppo"
TOTAL_TIMESTEPS = 2_000_000  # 200萬步通常能穩定通關
LEARNING_RATE = 2.5e-4
N_ENVS = 4                   # 同時開 4 個馬力歐 (Colab T4 極限)
SAVE_FREQ = 50_000           # 每 5 萬步存一次檔

# ================= Wrappers =================
class SkipFrame(gym.Wrapper):
    """
    關鍵加速：每 4 幀做一次決策，並加總獎勵。
    這讓模型能學到更長遠的後果，並提升 4 倍訓練速度。
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def make_env(env_id, rank, seed=0):
    """
    建立單一環境的工廠函數
    """
    def _init():
        env = gym_super_mario_bros.make(env_id)
        # 限制動作空間 (使用 SIMPLE_MOVEMENT 包含跑跳，足夠通關)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        # 1. Skip Frame (最重要)
        env = SkipFrame(env, skip=4)
        # 2. 灰階
        env = GrayScaleObservation(env, keep_dim=True)
        # 3. 縮放 (84x84)
        env = ResizeObservation(env, (84, 84))
        
        # Monitor 用於記錄每回合的 Reward，方便 TensorBoard 繪圖
        env = Monitor(env)
        
        env.seed(seed + rank)
        return env
    return _init

def main():
    # 確保儲存目錄存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 建立向量化環境 (8個平行環境)
    # SubprocVecEnv 會在不同 CPU 核心跑，大幅加速
    print(f"[INFO] Launching {N_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(ENV_ID, i) for i in range(N_ENVS)])
    
    # 2. Frame Stack (由 SB3 的 VecFrameStack 處理，更高效)
    # channels_order='last' -> (84, 84, 4) -> SB3 會自動轉成 (4, 84, 84) 給 CNN
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # 3. 定義模型
    # CnnPolicy: SB3 內建的 Nature CNN (跟 DeepMind 用的一樣)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=512,        # 每個環境跑 512 步更新一次 (共 512*8 = 4096 步)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/",
        device="cuda"       # 強制使用 GPU
    )

    # 4. 設定自動存檔 Callback
    # 注意：save_freq 是指每個環境的步數，所以實際儲存頻率是 SAVE_FREQ * N_ENVS
    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),
        save_path=SAVE_DIR,
        name_prefix="mario_ppo"
    )

    # 5. 開始訓練
    print("[INFO] Start training...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    
    # 6. 儲存最終模型
    model.save(os.path.join(SAVE_DIR, "mario_ppo_final"))
    print(f"[INFO] Model saved to {SAVE_DIR}")
    env.close()

if __name__ == "__main__":
    main()