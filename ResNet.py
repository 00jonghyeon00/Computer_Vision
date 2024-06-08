import torch
from torch import nn

class BasicBlock(nn.Module):
    
    # 클래스 속성
    expansion = 1

    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 3, stride = stride, padding = 1, bias = False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace = True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 3 , padding = 1, bias = False),
                                      nn.BatchNorm2d(inner_channels))                                  
        self.projection = projection
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):

        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x) # 점선
        else:
            shortcut = x # 실선
        
        out = self.relu(residual + shortcut)
        return out

class BottleNeck(nn.Module):


    # 클래스 속성
    expansion = 4

    def __init__(self, in_channels, inner_channels, stride = 1, projection = None):
        super().__init__()

        self.residual = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 1 ,bias = False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace = True),
                                      nn.Conv2d(inner_channels, inner_channels, 3, stride = stride, padding = 1, bias = False),
                                      nn.BatchNorm2d(inner_channels),
                                      nn.ReLU(inplace = True),
                                      nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias = False),
                                      nn.BatchNorm2d(inner_channels * self.expansion))
        
        self.projection = projection
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = self.residual(x)

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x
        
        out = self.relu(residual + shortcut)
        return out

class ResNet(nn.Module):
    
    def __init__(self, block, num_block_list, num_classes = 1000, zero_init_residual = True):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.stage1 = ~
        self.stage2 = ~
        self.stage3 = ~
        self.stage4 = ~
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    

