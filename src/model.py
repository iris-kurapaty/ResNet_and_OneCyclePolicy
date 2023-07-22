from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

def model_summary(Net, device):
  model = Net().to(device)
  return summary(model, input_size=(1, 28, 28))

class Net1(nn.Module):
  def __init__(self):
    super(Net1, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
    self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
    self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
    self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
    self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
    self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 |
    self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

  def forward(self, x):
    x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
    x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
    x = F.relu(self.conv6(F.relu(self.conv5(x))))
    # x = F.relu(self.conv7(x))
    x = self.conv7(x)
    x = x.view(-1, 10) #1x1x10> 10
    return F.log_softmax(x, dim=-1)

class Net2(nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 26; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 24; RF = 5
    
    # TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 12; RF = 6

    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 10; RF = 10

    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 8; RF = 14

    # TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 4 ; RF = 16

    # CONVOLUTION BLOCK 2
    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 2; RF = 24

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.ReLU()
    ) # output_size = 3; RF = 32

    # OUTPUT BLOCK
    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
    ) # output_size = 1

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.pool1(x)        
    x = self.convblock3(x)
    x = self.convblock4(x)
    x = self.pool2(x)             
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)    

class Net3(nn.Module):
  def __init__(self):
    super(Net3, self).__init__()
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 26; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 24; RF = 5
    
    # TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 12; RF = 6

    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.ReLU()
    ) # output_size = 12; RF = 10

    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 10; RF = 14

    # TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 5 ; RF = 16

    # CONVOLUTION BLOCK 2
    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.ReLU()
    ) # output_size = 5; RF = 24

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 3; RF = 32

    # OUTPUT BLOCK
    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
    ) # output_size = 1

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.pool1(x)        
    x = self.convblock3(x)
    x = self.convblock4(x)
    x = self.pool2(x)             
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)    


class Net4(nn.Module):
  def __init__(self):
    super(Net4, self).__init__()
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    ) # output_size = 26; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    ) # output_size = 24; RF = 5
    
    # TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 12; RF = 6

    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU()
    ) # output_size = 12; RF = 10

    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    ) # output_size = 10; RF = 14

    # TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 5 ; RF = 16

    # CONVOLUTION BLOCK 2
    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU()
    ) # output_size = 5; RF = 24

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    ) # output_size = 3; RF = 32

    # OUTPUT BLOCK
    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
    ) # output_size = 1

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.pool1(x)        
    x = self.convblock3(x)
    x = self.convblock4(x)
    x = self.pool2(x)             
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)   


class Net5(nn.Module):
  def __init__(self):
    super(Net5, self).__init__()
    drop_out = 0.1
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 26; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 24; RF = 5
    
    # TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 12; RF = 6

    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 12; RF = 10

    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 10; RF = 14

    # TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 5 ; RF = 16

    # CONVOLUTION BLOCK 2
    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 24

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 3; RF = 32

    # OUTPUT BLOCK
    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
    ) # output_size = 1

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.pool1(x)        
    x = self.convblock3(x)
    x = self.convblock4(x)
    x = self.pool2(x)             
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)   

class Net6(nn.Module):
  def __init__(self):
    super(Net6, self).__init__()
    drop_out = 0.1
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 26; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 24; RF = 5
    
    # TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 12; RF = 6

    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 12; RF = 10

    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 10; RF = 14

    # TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 5 ; RF = 16

    # CONVOLUTION BLOCK 2
    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 24

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 32

    # OUTPUT BLOCK
    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    ) # output_size = 5
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=5)
    ) # output_size = 1

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.pool1(x)        
    x = self.convblock3(x)
    x = self.convblock4(x)
    x = self.pool2(x)             
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = self.gap(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1) 

class Net7(nn.Module):
  def __init__(self):
    super(Net7, self).__init__()
    drop_out = 0.05
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 26; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 24; RF = 5

    # TRANSITION BLOCK 1
    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 24; RF = 7

    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 12; RF = 8


    # CONVOLUTION BLOCK 2
    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 10; RF = 12

    # TRANSITION BLOCK 2
    # self.pool2 = nn.MaxPool2d(2, 2)
    # # output_size =  ; RF =


    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 8; RF = 16

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 6; RF = 20

    # OUTPUT BLOCK
    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(8),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 4; ; RF = 24
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=4)
    ) # output_size = 1

    self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.convblock4(x)
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = self.gap(x)
    x = self.convblock8(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)
    
class BN_Net(nn.Module):
  def __init__(self):
    super(BN_Net, self).__init__()
    drop_out = 0.1
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 30; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 28; RF = 5

    # TRANSITION BLOCK 1
    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=1, bias=False),
      nn.ReLU()
    ) # output_size = 28; RF = 7

    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 14; RF = 8


    # CONVOLUTION BLOCK 2
    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 12; RF = 12

    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 10; RF = 16

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
      nn.BatchNorm2d(64),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 8; RF = 20


    #TRANSITION BLOCK 2

    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 8; RF = 24
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 4 ; RF = 26

    # CONVOLUTION BLOCK 3
    self.convblock8 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 4; RF = 34

    self.convblock9 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 4; RF = 42

    self.convblock10 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 4; RF = 50


    # OUTPUT BLOCK

    # output_size = 4; ; RF = 50
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=4)
    ) # output_size = 1

    self.convblock11 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.convblock4(x)
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = self.pool2(x)
    x = self.convblock8(x)
    x = self.convblock9(x)
    x = self.convblock10(x)
    x = self.gap(x)
    x = self.convblock11(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)

class LN_Net(nn.Module):
  def __init__(self):
    super(LN_Net, self).__init__()
    drop_out = 0.1
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 3), padding=0, bias=False),
      nn.LayerNorm([4, 30, 30]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 30; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), padding=0, bias=False),
      nn.LayerNorm([4, 28, 28]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 30; RF = 5

    # TRANSITION BLOCK 1
    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 30; RF = 7

    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 15; RF = 8


    # CONVOLUTION BLOCK 2
    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
      nn.LayerNorm([8, 14, 14]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 15; RF = 12

    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.LayerNorm([16, 14, 14]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 13; RF = 16

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.LayerNorm([16, 14, 14]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 11; RF = 20


    #TRANSITION BLOCK 2

    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 11; RF = 24
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 5 ; RF = 26

    # CONVOLUTION BLOCK 3
    self.convblock8 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.LayerNorm([16, 7, 7]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 34

    self.convblock9 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.LayerNorm([16, 7, 7]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 42

    self.convblock10 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
      nn.LayerNorm([32, 5, 5]),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 50


    # OUTPUT BLOCK

    # output_size = 5; ; RF = 50
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=5)
    ) # output_size = 1

    self.convblock11 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.convblock4(x)
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = self.pool2(x)
    x = self.convblock8(x)
    x = self.convblock9(x)
    x = self.convblock10(x)
    x = self.gap(x)
    x = self.convblock11(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1) 

class GN_Net(nn.Module):
  def __init__(self):
    super(GN_Net, self).__init__()
    drop_out = 0.1
    num_groups = 4
    # Input Block
    self.convblock1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
      nn.GroupNorm(num_groups, 16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 30; RF = 3

    # CONVOLUTION BLOCK 1
    self.convblock2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
      nn.GroupNorm(num_groups, 32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 30; RF = 5

    # TRANSITION BLOCK 1
    self.convblock3 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 30; RF = 7

    self.pool1 = nn.MaxPool2d(2, 2)
    # output_size = 15; RF = 8


    # CONVOLUTION BLOCK 2
    self.convblock4 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.GroupNorm(num_groups, 16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 15; RF = 12

    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
      nn.GroupNorm(num_groups, 32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 13; RF = 16

    self.convblock6 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
      nn.GroupNorm(num_groups, 64),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 11; RF = 20


    #TRANSITION BLOCK 2

    self.convblock7 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
      nn.ReLU()
    ) # output_size = 11; RF = 24
    self.pool2 = nn.MaxPool2d(2, 2)
    # output_size = 5 ; RF = 26

    # CONVOLUTION BLOCK 3
    self.convblock8 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
      nn.GroupNorm(num_groups, 16),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 34

    self.convblock9 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
      nn.GroupNorm(num_groups, 32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 42

    self.convblock10 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
      nn.GroupNorm(num_groups, 32),
      nn.Dropout(drop_out),
      nn.ReLU()
    ) # output_size = 5; RF = 50


    # OUTPUT BLOCK

    # output_size = 5; ; RF = 50
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=5)
    ) # output_size = 1

    self.convblock11 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.convblock4(x)
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.convblock7(x)
    x = self.pool2(x)
    x = self.convblock8(x)
    x = self.convblock9(x)
    x = self.convblock10(x)
    x = self.gap(x)
    x = self.convblock11(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1) 
    
def train(model, device, train_loader, optimizer, epoch):
  train_losses = []
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc = (100*correct/processed)

  return train_losses, train_acc


def test(model, device, test_loader):
  test_losses = []
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc = (100. * correct / len(test_loader.dataset))
  return test_losses, test_acc

  