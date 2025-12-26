import torch as T
import torch.nn.functional as F
import numpy as np
import random

class ReplayMemory:
    """
    使用預先分配的 Numpy Array 來儲存經驗，大幅減少記憶體碎片化與佔用。
    強制儲存 dtype=np.uint8，節省 4 倍記憶體。
    """
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 預先分配記憶體
        # state_shape 通常是 (4, 84, 84)
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        # 存入時保持 uint8 (0-255)，不做正規化
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

    def __len__(self):
        return self.size

class DQN:
    def __init__(self, model_cls, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_count = 0

        # 初始化網路
        self.q_net = model_cls(state_dim, action_dim).to(self.device)
        self.tgt_q_net = model_cls(state_dim, action_dim).to(self.device)
        self.tgt_q_net.load_state_dict(self.q_net.state_dict())
        self.tgt_q_net.eval() # Target net 不需要計算梯度

        self.optimizer = T.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state, deterministic=False):
        """
        Input state: np.array (4, 84, 84) uint8 [0-255] or float [0-1]
        """
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # 處理輸入：轉 Tensor -> Float -> Normalize -> Add Batch Dim -> GPU
        if isinstance(state, np.ndarray):
            state = T.tensor(state, dtype=T.float32, device=self.device)
        
        # 如果還是 0-255，就除以 255
        if state.max() > 1.0:
            state = state / 255.0
            
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with T.no_grad():
            q_values = self.q_net(state)
            return q_values.argmax().item()

    def train_per_step(self, state_dict):
        # 1. 取出數據並轉移到 GPU，同時做 Normalize
        states = T.tensor(state_dict['states'], dtype=T.float32, device=self.device) / 255.0
        next_states = T.tensor(state_dict['next_states'], dtype=T.float32, device=self.device) / 255.0
        actions = T.tensor(state_dict['actions'], dtype=T.int64, device=self.device).unsqueeze(1)
        rewards = T.tensor(state_dict['rewards'], dtype=T.float32, device=self.device)
        dones = T.tensor(state_dict['dones'], dtype=T.float32, device=self.device)

        # 2. 計算 Current Q (Q_eval)
        # gather: 選擇對應 action 的 Q 值
        q_eval = self.q_net(states).gather(1, actions).squeeze(1)

        # 3. 計算 Target Q (Double DQN)
        with T.no_grad():
            # 使用 Q_net 選擇動作 (解耦選擇與評估)
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            # 使用 Target_net 評估該動作的價值
            q_next = self.tgt_q_net(next_states).gather(1, next_actions).squeeze(1)
            
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # 4. Loss & Backprop
        loss = F.smooth_l1_loss(q_eval, q_target) # Huber Loss 比 MSE 更穩定
        
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        T.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 5. Update Target Network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.tgt_q_net.load_state_dict(self.q_net.state_dict())