import numpy as np
import os
import torch
from tqdm import tqdm

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import StepAPICompatibility, ResizeObservation, GrayScaleObservation, FrameStack

# 必須引入你的模型架構
from model import CustomCNN
from DQN import DQN

# ========== Config ===========
# 請確保這裡指向的是你「重新訓練後」的新模型 (4 channel 版本)
MODEL_PATH = os.path.join("ckpt_test", "best.pth") 

# 必須跟訓練時 (run.py) 的設定完全一致
ENV_ID = 'SuperMarioBros-1-1-v0'

# 定義簡化動作 (跟 run.py 一模一樣)
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
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
    env = StepAPICompatibility(env, new_step_api=False)

    # 設定動作空間
    movement = REDUCED_MOVEMENT if USE_REDUCED_ACTIONS else SIMPLE_MOVEMENT
    env = JoypadSpace(env, movement)

    # === 關鍵修改：加入跟訓練時一樣的 Wrappers ===
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=4)  # 變成 (4, 84, 84)
    # ==========================================
    
    return env

# ... Config ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【修改】輸入通道變為 4
OBS_SHAPE = (4, 84, 84) 

# 計算動作數量
dummy_env = make_env()
N_ACTIONS = dummy_env.action_space.n
dummy_env.close()

VISUALIZE = True
TOTAL_EPISODES = 10
MAX_STEPS = 5000  # 防止卡死

# ========== Initialize DQN =========== 
dqn = DQN( 
    model=CustomCNN, 
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,  
    gamma=0.99,          
    epsilon=0.0,
    target_update=1000,
    device=device
)

# ========== 載入模型權重 =========== 
if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ========== Evaluation Loop ===========
env = make_env()

for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset() # 這裡出來已經是 LazyFrames (4, 84, 84)

    # 【修改】處理 State：轉為 numpy -> float -> normalize -> 增加 batch 維度
    state = np.array(state).astype(np.float32) / 255.0
    state = np.expand_dims(state, axis=0) # (1, 4, 84, 84)

    done = False
    total_reward = 0
    steps = 0
    
    # 增加卡關檢測 (避免負分刷到底)
    prev_x = 0
    stagnation = 0
    
    while not done and steps < MAX_STEPS:
        # 轉 Tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            action_probs = torch.softmax(dqn.q_net(state_tensor), dim=1)
            action = torch.argmax(action_probs, dim=1).item()

        next_state, reward, done, info = env.step(action)

        # 【修改】處理 Next State
        next_state = np.array(next_state).astype(np.float32) / 255.0
        next_state = np.expand_dims(next_state, axis=0) # (1, 4, 84, 84)

        total_reward += reward
        state = next_state
        steps += 1

        # 簡單的卡關檢測 (如果在同一個 X 座標卡太久就強制結束)
        x_pos = info.get("x_pos", 0)
        if x_pos == prev_x:
            stagnation += 1
            if stagnation > 500: # 卡 500 步就結束
                done = True
        else:
            stagnation = 0
        prev_x = x_pos

        if VISUALIZE:
            env.render()

    print(f"Episode {episode}/{TOTAL_EPISODES} - Total Reward: {total_reward:.2f} - Steps: {steps}")

env.close()