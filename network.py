import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.linear2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, output_size).to(self.device)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        if x.dim() == 1:
            x = torch.unsqueeze(x,0)
        x = x.to(self.device)
        x = self.bn1(F.relu(self.linear1(x)))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, output_size=1):
        super(Critic, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size+action_size, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, output_size).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = self.bn1(F.relu(self.linear1(state)))
        x = torch.cat((x, action.float()), dim=1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

