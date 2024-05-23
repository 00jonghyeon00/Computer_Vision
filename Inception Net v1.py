import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                                        nn.BatchNorm2d(out_channels, eps=0.001),
                                        nn.ReLU())
        
    def forward(self, x):
        x = self.conv_block(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj): # red는 reduce를 줄인 말
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                                     BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1))
        
        self.branch3 = nn.Sequential(BasicConv2d(in_channels, ch5x5red, kernel_size=1),
                                     BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     BasicConv2d(in_channels, pool_proj, kernel_size=1))
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)   
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, drop_p = 0.7):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux 1 : N x 512 x 14 x 14
        # aux 2 : N x 528 x 14 x 14
        # 자세한 사항은 Inception Net v1 표 참고
        x = self.avgpool(x)
        
        # aux 1 : N x 512 x 4 x 4
        # aux 2 : N x 528 x 4 x 4
        x = self.conv(x)

        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)

        # N x 2048
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        # N x 1024
        x = self.fc2(x)

        # N x 1000(num_classes)
        return x



