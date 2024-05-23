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

class Inception_V1(nn.Module):
    def __init__(self, num_classes = 1000, use_aux=True, init_weights=None, drop_p=0.4, drop_p_aux=0.7):
        super().__init__()

        self.use_aux = use_aux

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2a = BasicConv2d(64, 64, kernel_size=1)
        self.conv2b = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        if use_aux:
            self.aux1 = InceptionAux(512, num_classes, drop_p=drop_p_aux)
            self.aux2 = InceptionAux(528, num_classes, drop_p=drop_p_aux)
        else:
            self.aux1 = None
            self.aux2 = None
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # GAP
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
    
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2a(x)
        # N x 64 x 56 x 56
        x = self.conv2b(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)

        # N x 512 x 14 x 14
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None  # 뭐라도 넣어놔야해서 not defined error

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)

        # N x 528 x 14 x 14
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

# x, aux2, aux1 = model(torch.randn(2,3,224,224))
# print(x.shape)
# print(aux2.shape)
# print(aux1.shape)



