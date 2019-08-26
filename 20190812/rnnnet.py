import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from MyData import MyDataset

img_path = "code"
BATCH_SIZE = 64
NUM_WORKERS = 1
EPOCH = 100
save_path = r'data/test.pkl'

class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180, 128),  # [batch_size*120,128]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        self.out = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(-1, 180, 120).permute(0, 2, 1)  # [batch_size,120,180]
        x = x.contiguous().view(-1, 180)  # [batch_size*120,180]
        fc1 = self.fc1(x)  # [batch_size*120,128]
        fc1 = fc1.view(-1, 120, 128)  # [batch_size,120,128]
        lstm, (h_n, h_c) = self.lstm(fc1)  # [batch_size,120,128]
        out = lstm[:, -1, :]

        out = out.view(-1,1,128)
        out = out.expand(BATCH_SIZE,4,128)             # [batch_size,4,128]
        lstm,(h_n,h_c) = self.lstm(out)         # [batch_size,4,128]
        y1 = lstm.contiguous().view(-1,128)        # [batch_size*4,128]
        out = self.out(y1)                         # [batch_size*4,10]
        out = out.view(-1,4,10)                    # [batch_size,4,10]
        output = F.softmax(out,dim=2)

        return out,output

if __name__ == '__main__':
    net = RnnNet()
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))

    train_data = MyDataset(root="code")
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=NUM_WORKERS)

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            batch_x = x
            batch_y = y.float()

            decoder = net(batch_x)
            out,output = decoder[0], decoder[1]
            loss = loss_func(out,batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                test_y = torch.argmax(y,2).data.numpy()
                pred_y = torch.argmax(output,2).cpu().data.numpy()
                accuracy = np.mean(np.all(pred_y==test_y,axis=1))
                print("epoch:", epoch, "  |  ", "i:", i, "  |  ", "loss:", '%.4f' % loss.item(), "  |  ", 'accuracy:', "%.2f%%" % (accuracy * 100))
                print("test_y:",test_y[0])
                print("pred_y:",pred_y[0])

        torch.save(net.state_dict(), save_path)