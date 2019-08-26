import torch
import torch.nn as nn
import numpy as np
from test01 import get_data
from test02 import Net

path = r"./data1.xls"
net = Net()
# net.load_state_dict(torch.load("./params"))
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters())
datas = get_data.red_excel(path)
max_data = np.max(datas)
train_data = np.array(datas)/max_data
print(train_data)

for epoch in range(10):
    for i in range(len(train_data)-9):
        x = train_data[i:i+9]
        y = train_data[i+9:i+10]
        xs = torch.reshape(torch.tensor(x,dtype=torch.float32),[-1,1,9])
        ys = torch.tensor(y,dtype=torch.float32)
        _y = net(xs)
        loss = loss_fn(_y,ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        out = int(_y*max_data)
        label = int(ys*max_data)
        print(out)
        print(label)
        print(i,"/",epoch)
    torch.save(net.state_dict(),"./params.pth")