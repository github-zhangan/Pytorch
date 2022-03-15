import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./pytorch_learning/data",train= False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=64)


class Tudui(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels = 3,out_channels = 6,kernel_size = 3,stride = 1,padding = 0)


    def forward(self,x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("logs")
tudui = Tudui()
step = 0
for data in dataloader:
    img, target = data
    output= tudui(img)

    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("input",img,step)
    writer.add_images("output",output,step)
    step = step +1

print(tudui)

查看命令
tensorboard --logdir=一定要绝对路径
