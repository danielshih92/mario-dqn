import os
import numpy as np
import torch
from tqdm import tqdm
import glob
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from reward import shaped_reward
from model import CustomCNN
from DQN import DQN, ReplayMemory

# ===================== Config =====================
ENV_ID = "SuperMarioBros-1-1-v0"
LR = 0.00025               # 稍微調高一點點
BATCH_SIZE = 32            # 32 或 64 都可以
GAMMA = 0.99
MEMORY_SIZE = 50_000       # 5萬 frames 對 Mario 夠用了，且絕對不會爆 RAM
TARGET_UPDATE = 1000      
TOTAL_EPISODES = 5000

EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY_EPISODES = 2000 

SAVE_DIR = "/content/drive/MyDrive/mario/ckpt"
SAVE_EVERY = 200
EVAL_INTERVAL = 100
MAX_EVAL_STEPS = 3000

RESUME = False
RESUME_PATH = "" 

USE_REDUCED_ACTIONS = True
# 稍微優化的動作空間：移除單純的 "A" (跳)，通常我們希望向右跑並跳
REDUCED_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left']
]

# ===================== Custom Wrappers =====================
class SkipFrame(gym.Wrapper):
    """每做一次決策，重複執行 skip 次，合併獎勵。大幅提升速度。"""
    def __init__(self, env, skip):
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

def make_env():
    env = gym_super_mario_bros.make(ENV_ID)
    env = JoypadSpace(env, REDUCED_MOVEMENT if USE_REDUCED_ACTIONS else SIMPLE_MOVEMENT)
    
    # 關鍵：加入 SkipFrame (通常是 4)
    env = SkipFrame(env, skip=4)
    
    # 影像處理管道
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env, keep_dim=False)
    # Transform to Normalize not here! Keep it uint8 until DQN.
    env = FrameStack(env, num_stack=4)
    return env

def epsilon_by_episode(ep):
    return max(EPS_END, EPS_START - (ep / EPS_DECAY_EPISODES) * (EPS_START - EPS_END))

@torch.no_grad()
def run_eval(dqn, ep):
    env = make_env()
    state = env.reset() # LazyFrames (4, 84, 84) uint8
    done = False
    total_reward = 0
    steps = 0
    
    # Evaluation 時不需要 shaped reward，看原始分數即可
    while not done and steps < MAX_EVAL_STEPS:
        # LazyFrames -> np.array (uint8)
        state_np = np.array(state) 
        action = dqn.take_action(state_np, deterministic=True)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    env.close()
    return total_reward, steps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = make_env()
    # 測試一下 env 輸出形狀
    tmp_obs = env.reset()
    obs_shape = np.array(tmp_obs).shape # 預期 (4, 84, 84)
    n_actions = env.action_space.n
    
    print(f"Observation shape: {obs_shape}, Action space: {n_actions}")

    dqn = DQN(
        model_cls=CustomCNN,
        state_dim=obs_shape,
        action_dim=n_actions,
        learning_rate=LR,
        gamma=GAMMA,
        epsilon=EPS_START,
        target_update=TARGET_UPDATE,
        device=device,
    )

    if RESUME and os.path.exists(RESUME_PATH):
        dqn.q_net.load_state_dict(torch.load(RESUME_PATH))
        print("Model loaded.")

    # 初始化 Memory，注意這裡傳入 shape
    memory = ReplayMemory(MEMORY_SIZE, obs_shape)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_score = -999999

    for ep in tqdm(range(1, TOTAL_EPISODES + 1)):
        state = env.reset() # LazyFrames uint8
        done = False
        
        # 轉換為 numpy uint8 以便處理，但在記憶體中我們已經優化
        state_np = np.array(state, dtype=np.uint8)
        
        dqn.epsilon = epsilon_by_episode(ep)
        
        prev_info = {"x_pos": 0, "coins": 0, "score": 0, "life": 2}
        stagnation_counter = 0
        ep_reward = 0
        
        while not done:
            # 選擇動作
            action = dqn.take_action(state_np, deterministic=False)
            
            # 執行動作
            next_state, reward, done, info = env.step(action)
            next_state_np = np.array(next_state, dtype=np.uint8)
            
            # 處理 Reward Shaping
            # 計算 stagnation
            x_pos = info.get("x_pos", 0)
            if x_pos == prev_info.get("x_pos", 0):
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # 使用你的 reward.py
            r_shaped = shaped_reward(info, reward, prev_info, stagnation_counter)
            prev_info = info
            
            # 存入記憶體 (存 uint8)
            memory.push(state_np, action, r_shaped, next_state_np, done)
            
            state_np = next_state_np
            ep_reward += reward

            # 開始訓練
            if len(memory) > 2000: # 稍微累積一點資料再練
                batch = memory.sample(BATCH_SIZE)
                state_dict = {
                    "states": batch[0],
                    "actions": batch[1],
                    "rewards": batch[2],
                    "next_states": batch[3],
                    "dones": batch[4],
                }
                dqn.train_per_step(state_dict)

        # Log
        if ep % 10 == 0:
            print(f"Ep {ep} | Reward: {ep_reward:.1f} | Epsilon: {dqn.epsilon:.3f} | Mem: {len(memory)}")

        # Save & Eval
        if ep % SAVE_EVERY == 0:
            torch.save(dqn.q_net.state_dict(), os.path.join(SAVE_DIR, "latest.pth"))
        
        if ep % EVAL_INTERVAL == 0:
            score, steps = run_eval(dqn, ep)
            print(f"[EVAL] Ep {ep} Score: {score:.1f}, Steps: {steps}")
            if score > best_score:
                best_score = score
                torch.save(dqn.q_net.state_dict(), os.path.join(SAVE_DIR, "best.pth"))
                print(f"New Best Model Saved! Score: {score}")

    env.close()

if __name__ == "__main__":
    main()