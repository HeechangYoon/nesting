import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, state_size, x_action_size, y_action_size, a_action_size):
        super(Network, self).__init__()
        self.state_size = state_size
        self.x_action_size = x_action_size
        self.y_action_size = y_action_size
        self.a_action_size = a_action_size

        self.conv1 = nn.Conv2d(in_channels=state_size[-1], out_channels=16, kernel_size=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=state_size[-1], out_channels=32, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(32 * 52 * 11, 128)
        self.fc_x_pi = nn.Linear(128, x_action_size)
        self.fc_y_pi = nn.Linear(128, y_action_size)
        self.fc_a_pi = nn.Linear(128, a_action_size)
        self.fc_v = nn.Linear(128, 1)

    def x_pi(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        # x = x.squeeze(2)
        # x = F.relu(self.conv2(x))
        x = x.contiguous().view(-1, x.size(0))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc_x_pi(x)
        return x

    def y_pi(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        # x = x.squeeze(2)
        # x = F.relu(self.conv2(x))
        x = x.contiguous().view(-1, x.size(0))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc_y_pi(x)
        return x

    def a_pi(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        # x = x.squeeze(2)
        # x = F.relu(self.conv2(x))
        x = x.contiguous().view(-1, x.size(0))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc_a_pi(x)
        return x

    def v(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        # x = x.squeeze(2)
        # x = F.relu(self.conv2(x))
        x = x.contiguous().view(-1, x.size(0))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        v = self.fc_v(x)
        return v

    def get_action(self, s, possible_actions):
        s = torch.from_numpy(s).float().to(device).unsqueeze(0)
        logit = self.pi(s.permute(0, 3, 1, 2))
        mask = np.ones(self.action_size)
        mask[possible_actions] = 0.0
        logit = logit - 1e8 * torch.from_numpy(mask).float().to(device)
        prob = torch.softmax(logit, dim=-1)[0]

        m = Categorical(prob)
        a = m.sample().item()

        return a