import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

model = Model()
print(model)
x = torch.tensor(1)
x = torch.reshape(x,(1,1,1))
print(x.shape)
output = model(x)
print(output)

123456
