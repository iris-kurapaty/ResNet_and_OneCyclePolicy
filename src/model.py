from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

# dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Prep Layer input 32/1/1
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size =32

        # Layer 1
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            
        ) # output_size = 32

        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
            ) # output_size = 16

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # output_size = 16

        # Layer 3
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            
        ) # output_size = 8

        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
            ) # output_size = 8

        # MAx Pooling with Kernel Size 4
        self.maxpool =  nn.MaxPool2d(4, 2)


        # fully connected layer
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.preplayer(x)
        x = self.convlayer1(x)
        r1 = self.res1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.convlayer2(x)
        r2 = self.res2(x)
        x = x + r2
        x = self.maxpool(x)
        x = self.fc(torch.squeeze(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Block(nn.Module):
  def __init__(self, input_size, out_size, drop_out):
    super(Block, self).__init__()
    self.drop_out = drop_out

    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=input_size, out_channels=out_size, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 3

    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=out_size, out_channels = out_size, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 5

    self.convblock3 = nn.Sequential(
        nn.Conv2d(out_size, out_size, kernel_size = (3,3), padding=1, dilation = 2, stride=2, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 9

  def __call__(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x) 
    x = self.convblock3(x) 
    
    return x

class DepthWiseConvolution(nn.Module):
  def __init__(self, input_size, output_size):
    super(DepthWiseConvolution, self).__init__()

    self.depthwise1 = nn.Sequential(
        nn.Conv2d(input_size, input_size, kernel_size = (3,3),padding= 1,groups = input_size),
        nn.ReLU()
    )
    self.pointwise1 =  nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size = (1,1)),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
    self.depthwise2 = nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (3,3),padding= 1,groups = output_size),
        nn.ReLU()
    )
    self.pointwise2 =  nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (1,1)),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
    self.depthwise3 = nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (3,3),padding= 1,groups = output_size),
        nn.ReLU()
    )
    
    self.pointwise3 =  nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (1,1), padding= 0),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
   

  def __call__(self, x):
    x = self.depthwise1(x)
    x = self.pointwise1(x)
    x = self.depthwise2(x)
    x = self.pointwise2(x)    
    x = self.depthwise3(x)
    x = self.pointwise3(x)
    return x

# Block 1: 3, 5, 9    
# Block 2: 13, 17, 25
# Block 3: 25, 33, 41
# Block 4: 49, 57, 65

class Net_9(nn.Module):
  def __init__(self, drop_out = 0.1):
    super(Net, self).__init__()
    self.drop_out = drop_out

    # Input Block + Convolution Blocks
    self.layer1 = Block(3, 32, 0.1)
    self.layer2 = Block(32, 64, 0.1)

    # Depth-Wise Separable Convolutions
    self.layer3 = DepthWiseConvolution(64, 128)

   # OUTPUT BLOCK

    # output_size = 4; ; RF = 50
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=7)
    ) # output_size = 1

    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.gap(x)
    x = self.convblock5(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)