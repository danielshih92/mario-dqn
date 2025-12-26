import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        # store as numpy arrays / python scalars
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states, axis=0)       # (B,C,84,84)
        next_states = np.stack(next_states, axis=0)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    Double DQN + Dueling network + Huber loss + grad clip
    """
    def __init__(
        self,
        model_cls,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        target_update_steps=5000,
        device="cuda",
        grad_clip=10.0,
    ):
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.target_update_steps = int(target_update_steps)
        self.grad_clip = float(grad_clip)

        self.policy_net = model_cls(state_dim, action_dim).to(self.device)
        self.target_net = model_cls(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.global_step = 0
        self.epsilon = 1.0  # set by training loop

    @torch.no_grad()
    def act(self, state, deterministic=False):
        if (not deterministic) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.action_dim)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.policy_net(s)  # (1,A)
        return int(torch.argmax(q, dim=1).item())

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q(s,a)
        q_sa = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN target:
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states_t), dim=1, keepdim=True)  # (B,1)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optim.step()

        self.global_step += 1
        if self.global_step % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path, map_location=None):
        if map_location is None:
            map_location = self.device
        state = torch.load(path, map_location=map_location)
        self.policy_net.load_state_dict(state)
        self.target_net.load_state_dict(self.policy_net.state_dict())
