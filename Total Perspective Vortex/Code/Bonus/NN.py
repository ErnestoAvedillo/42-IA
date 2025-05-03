
import torch, torch.nn as nn

class SmallCSPCNN(nn.Module):
    def __init__(self, n_components, n_times, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_components, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)
