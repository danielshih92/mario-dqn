import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    經典的 Nature CNN 架構 (來自 DQN 2015 論文)。
    對於 84x84 的 Atari/Mario 遊戲，這通常比 ResNet 更快且更容易收斂。
    """
    def __init__(self, input_shape, num_actions):
        super(CustomCNN, self).__init__()
        
        c, h, w = input_shape
        
        self.features = nn.Sequential(
            # Conv 1: 32 filters, 8x8 kernel, stride 4
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Conv 2: 64 filters, 4x4 kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv 3: 64 filters, 3x3 kernel, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 計算卷積後的特徵圖大小
        # 84x84 -> (stride 4) -> 21x21 -> (stride 2) -> 11x11 -> (stride 1) -> 7x7 (Padding=0時約略值，需精確計算)
        # 實際計算: 
        # 84 -> (84-8)/4 +1 = 20
        # 20 -> (20-4)/2 +1 = 9
        # 9  -> (9-3)/1 +1  = 7
        # 最終是 64 * 7 * 7 = 3136
        
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

