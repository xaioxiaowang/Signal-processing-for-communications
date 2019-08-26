import torch.nn as nn
import torch

batch = 1000
class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer = nn.GRU(28,64,1,batch_first=True)
        self.output_layer = nn.Linear(64,10)
    def forward(self, x):
        #NCHW-->NS(C*H)V(W)
        input = x.view(-1,28,28)

        h0 = torch.zeros(1,batch,64)
        # c0 = torch.zeros(1, batch, 64)

        outputs,hn = self.rnn_layer(input,h0)

        output = outputs[:,-1,:]#只要NSV的最后一个S的数据
        output = self.output_layer(output)
        return output