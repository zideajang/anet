import numpy as np
from scipy import signal
from layer import Layer

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=None)

mnist_testset = datasets.MNIST(root='./data', train=False, download=False, transform=None)
print(len(mnist_testset))
print(len(mnist_trainset))

print(type(mnist_trainset))

# help(torchvision.datasets.mnist.MNIST)

class Conv2d(Layer):

    def __init__(self,input_size,in_channel,out_channel,kernel_size) -> None:
        super().__init__()

        self.input_width  = input_size[0]
        self.input_height = input_size[1]

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size

        self.kernels
        self.biases = np.random.randn()
        
    def forward(self,input):
        self.input = input

        return self.output

    # learning rate
    def backword(self,ouput_grad,lr):
        pass
        

def data_process(x,y,limit):
    pass

# 分析数据机
img = mnist_trainset[0][0]
print(type(img))
label = mnist_trainset[0][1]
print(label)
print(np.asarray(img))
# train_dataloader = DataLoader(mnist_trainset,batch_size=12)
# for data in train_dataloader:
#     print(data)
#     break

# 定义网络


# 定义训练


if __name__ == "__main__":
    pass
