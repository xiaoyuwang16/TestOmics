import torch
import torch.nn as nn
import torch.nn.init as init

class TestOmix(nn.Module):
    def __init__(self, In_Nodes, dropout_rate1, dropout_rate2, dim1, dim2):
        super(TestOmix, self).__init__()

        self.sc1 = nn.Linear(In_Nodes, dim1)
        self.sc2 = nn.Linear(dim1, dim2)
        self.sc3 = nn.Linear(dim2, 2, bias=True)
        self.sc4 = nn.Linear(2, 1, bias=True)

        init.xavier_uniform_(self.sc1.weight)
        init.xavier_uniform_(self.sc2.weight)
        init.xavier_uniform_(self.sc3.weight)
        init.xavier_uniform_(self.sc4.weight)

        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sc1(x)
        x = self.tanh(x)
        x = self.dropout1(x)

        x = self.sc2(x)
        x = self.tanh(x)
        x = self.dropout2(x)

        x = self.sc3(x)
        x = self.tanh(x)
        x = self.sc4(x)

        return x

def reset_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()