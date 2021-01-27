import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelingDeepQNetwork(nn.Module):
    def __init__ (self, lr, n_actions, input_dims, fc1_dims, name, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims

        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.V = nn.Linear(self.fc1_dims, 1)
        self.A = nn.Linear(self.fc1_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))



