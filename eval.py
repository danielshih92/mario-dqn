import numpy as np
import torch
from tqdm import tqdm

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from model import CustomCNN
from DQN import DQN

# ========== Config ===========
MODEL_PATH = os.path.join("ckpt_test","step_18_reward_536_custom_586.pth")        # 模型權重檔案的存放路徑

#env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')                     # 建立《超級瑪利歐兄弟》的遊戲環境(第1個世界的第1關)

# SIMPLE_MOVEMENT可自行定義 以下為自訂範例:
# SIMPLE_MOVEMENT = [
#    # ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#    # ["A"],          # Jump straight up.
#     ["left"],       # Move left.
#     ["left", "A"], # Move right and jump.
#     ["left", "B"], # Move right and run.
#     ["left", "A", "B"], # Move right, run, and jump.
# ]

#env = JoypadSpace(env, SIMPLE_MOVEMENT) 

import gym
from gym.wrappers import StepAPICompatibility

# 1) make（這裡可能會自動包 TimeLimit）
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 2) 🔑 拆掉 TimeLimit（不拆一定炸 expected 5 got 4）
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) 固定成舊 step API（回 4-tuple）
env = StepAPICompatibility(env, output_truncation_bool=False)

# 4) 再包 JoypadSpace
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # 檢查是否有可用的 GPU，否則使用 CPU 作為運算設備
OBS_SHAPE = (1, 84, 84)                                                     # 遊戲畫面轉換為 (1, 84, 84) 的灰階圖像
N_ACTIONS = len(SIMPLE_MOVEMENT) 

VISUALIZE = True                                                            # 是否在每回合中顯示遊戲畫面
TOTAL_EPISODES = 10                                                         # 測試回合的總數

# ========== Initialize DQN =========== 
dqn = DQN( 
    model=CustomCNN, 
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,  
    gamma=0.99,          
    epsilon=0.0,                   # 設為 0.0 表示完全利用當下的策略
    target_update=1000,            # target [Q-net] 更新的頻率
    device=device
)

# ========== 載入模型權重 =========== 
if os.path.exists(MODEL_PATH):
    try:                                                                  # 檢查模型檔案是否存在：
        model_weights = torch.load(MODEL_PATH, map_location=device)       #  若存在，嘗試載入模型權重
        dqn.q_net.load_state_dict(model_weights)                          #    載入成功，應用到模型
        dqn.q_net.eval()                                                  #    載入失敗，輸出具體的錯誤資訊(錯誤資訊存在e中)
        print(f"Model loaded successfully from {MODEL_PATH}")             #  若不存在，則FileNotFoundError
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ========== Evaluation Loop ===========
for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset()                                                   # 重置環境到初始狀態，並獲取環境的 state 初始值
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)                                 # 新增 channel dimension ( [H, W] to [1, H, W] )
    state = np.expand_dims(state, axis=0)                                 # 新增 batch dimension ( [1, H, W] to [1, 1, H, W] )
                                                                          # 符合 CNN 輸入要求：[batch, channels, height, width]
    done = False
    total_reward = 0

    while not done:
        # Take action using the trained policy
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)    # 將 NumPy 格式的 state 轉換為 PyTorch 的 tensor 格式
        with torch.no_grad():                                                       
            action_probs = torch.softmax(dqn.q_net(state_tensor), dim=1)          # 使用訓練好的 [Q-net] 計算當前狀態的動作分數，並透過 Softmax 轉換為動作機率分佈，輸出範圍為[0,1]，總合為1            
                                                                                                                                            
            action = torch.argmax(action_probs, dim=1).item()                     # 選擇機率最高的動作作為當下策略的 action
        next_state, reward, done, info = env.step(action)                         # 根據選擇的 action 與環境互動，獲取 next_state、reward、是否終止

        # Preprocess next state
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)                           # 新增 channel dimension
        next_state = np.expand_dims(next_state, axis=0)                           # 新增 batch dimension

        # Accumulate rewards
        total_reward += reward
        state = next_state

        if VISUALIZE:                                                             # 如果 VISUALIZE=True，則用 env.render() 顯示環境當下的 state
            env.render()

    print(f"Episode {episode}/{TOTAL_EPISODES} - Total Reward: {total_reward}")   # 印出當下的進度 episode/總回合數 和該回合的 total_reward

env.close()
