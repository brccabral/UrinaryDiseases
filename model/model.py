import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear1H(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = self.dropout(output)
        output = self.fc2(output)
        return self.norm(output)
