import gym
import matplotlib.pyplot as plt
from pong_utils import preprocess_single, preprocess_batch, play
from torch import nn
import torch.nn.functional as F
from torch import optim


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(2, 2), stride=(2, 2))
        # 80 * 80 -> 40 * 40
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 2), stride=(2, 2))
        # 40 * 40 -> 20 * 20
        self.size = 20 * 20 * 8

        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 40 * 40 * 4
        x = F.relu(self.conv2(x))  # 20  * 20 * 4

        x = x.view(-1, self.size)

        x = self.fc1(x)
        x = self.fc2(x)

        out = self.sig(x)

        return out


env = gym.make('Pong-v0')
state = env.reset()

for _ in range(20):
    frame0, _, _, _ = env.step(env.action_space.sample())
    frame1, _, _, _ = env.step(env.action_space.sample())

frames = preprocess_batch([frame0, frame1])

policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=1e-3)
