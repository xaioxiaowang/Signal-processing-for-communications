import torch.nn as nn
import torch

batch = 1000
class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer = nn.GRUCell(28,64)
        self.output_layer = nn.Linear(64,10)
    def forward(self, x):
        #NCHW-->NS(C*H)V(W)
        input = x.view(-1,28,28)

        hx = torch.zeros(batch,64)

        for s in range(28):
            input = x[:,s,:]
            hx = self.rnn_layer(input,hx)

        output = self.output_layer(hx)
        return output