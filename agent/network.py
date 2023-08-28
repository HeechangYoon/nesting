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

        self.conv1 = nn.Conv2d(in_channels=state_size[-1], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1_x = nn.Linear(32 * 52 * 11, 1024)
        self.fc2_x = nn.Linear(1024, 512)
        self.fc3_x = nn.Linear(512, 256)
        self.fc1_y = nn.Linear(32 * 52 * 11, 1024)
        self.fc2_y = nn.Linear(1024, 512)
        self.fc3_y = nn.Linear(512, 256)
        self.fc1_a = nn.Linear(32 * 52 * 11, 1024)
        self.fc2_a = nn.Linear(1024, 512)
        self.fc3_a = nn.Linear(512, 256)
        self.fc1_v = nn.Linear(32 * 52 * 11, 1024)
        self.fc2_v = nn.Linear(1024, 512)
        self.fc3_v = nn.Linear(512, 256)
        self.fc_x_pi = nn.Linear(256, x_action_size)
        self.fc_y_pi = nn.Linear(256, y_action_size)
        self.fc_a_pi = nn.Linear(256, a_action_size)
        self.fc_v = nn.Linear(256, 1)

    def x_pi(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = x.contiguous().view(-1, 32 * 52 * 11)
        x = F.leaky_relu(self.fc1_x(x))
        x = F.leaky_relu(self.fc2_x(x))
        x = F.leaky_relu(self.fc3_x(x))
        x = self.fc_x_pi(x)
        return x

    def y_pi(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = x.contiguous().view(-1, 32 * 52 * 11)
        x = F.leaky_relu(self.fc1_y(x))
        x = F.leaky_relu(self.fc2_y(x))
        x = F.leaky_relu(self.fc3_y(x))
        x = self.fc_y_pi(x)
        return x

    def a_pi(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = x.contiguous().view(-1, 32 * 52 * 11)
        x = F.leaky_relu(self.fc1_a(x))
        x = F.leaky_relu(self.fc2_a(x))
        x = F.leaky_relu(self.fc3_a(x))
        x = self.fc_a_pi(x)
        return x

    def v(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = x.contiguous().view(-1, 32 * 52 * 11)
        x = F.leaky_relu(self.fc1_v(x))
        x = F.leaky_relu(self.fc2_v(x))
        x = F.leaky_relu(self.fc3_v(x))
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