import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])


class MyDataset(data.Dataset):
    def __init__(self,root):
        self.transform = data_transforms
        self.list = []

        for filenames in os.listdir(root):
            x = os.p.ath.join(root,filenames)
            ys = filenames.split('.')
            y = self.one_hot(ys[0])
            self.list.append([x,np.array(y)])

    def __len__(self):
        return len(self.list)
    def __getitem__(self, index):
        img_path,label = self.list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        label = torch.from_numpy(label)

        return img,label
    def one_hot(self,x):
        z = np.zeros(shape=[4,10])
        for i in range(4):
            index = int(x[i])
            z[i][index] +=1
        return z

if __name__ == '__main__':
    mydata = MyDataset("code")
    data_loader = data.DataLoader(dataset=mydata,batch_size=100,shuffle=True)
    for i,(x,y) in enumerate(data_loader):
        print(i)
        print(x.size())
        print(y.size())