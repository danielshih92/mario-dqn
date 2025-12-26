import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


class DuelingCNN(nn.Module):
    """
    Dueling DQN network:
      Q(s,a) = V(s) + (A(s,a) - mean(A))
    Input:  (B, C, 84, 84) where C = stack_k (default 4)
    Output: (B, num_actions)
    """
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape
        assert (h, w) == (84, 84), "Expected input 84x84"

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)  # 84 -> 20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1) # 20 -> 18

        # 18*18*64 = 20736
        self.fc = nn.Linear(18 * 18 * 64, 512)

        self.adv = nn.Linear(512, num_actions)
        self.val = nn.Linear(512, 1)

        self.apply(init_weights)

    def forward(self, x):
        # x: (B,C,84,84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        adv = self.adv(x)          # (B,A)
        val = self.val(x)          # (B,1)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q


