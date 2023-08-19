import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel*self.expansion:
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, model_type, num_classes, mode):  # model_type, num_classes, mode
        super(ResNet, self).__init__()
        self.cfg = {
            "ResNet18": [2, 2, 2, 2],
            "ResNet34": [3, 4, 6, 3],
            'ResNet50': [3, 4, 6, 3],
            'ResNet101': [3, 4, 23, 3],
            'ResNet152': [3, 8, 36, 3]
        }
        self.in_channel = 64
        self.model_type = model_type
        self.mode = mode
        self.block = BasicBlock if self.model_type in ["ResNet18", "ResNet34"] else Bottleneck
        self.blocks_num = self.cfg[self.model_type]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    
        self.layer1 = self._make_layer(self.block, 64, self.blocks_num[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.blocks_num[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.blocks_num[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.blocks_num[3], stride=2)
        self.projection = self._get_projection(self.mode)
        self.linear1 = nn.Linear(512*self.block.expansion, num_classes)
        self.linear2 = nn.Linear(512*self.block.expansion, num_classes)

    def _get_projection(self, mode):
        if mode == "MaxPool":
            return nn.Sequential(nn.MaxPool2d(2, 2))
        elif mode == "AvgPool":
            return nn.Sequential(nn.AvgPool2d(2, 2))
        elif self.mode == "ConvOne":
            return nn.Sequential(nn.Conv2d(256*self.block.expansion, 256*self.block.expansion, 1, 1, 0, bias=False))

    def _make_layer(self, block, channel, block_num, stride):
        strides = [stride] + [1] * (block_num - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channel, stride=stride))
            self.in_channel = channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.mode == "default":
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x1 = self.linear1(x)
            x2 = self.linear2(x)
            return x1, x2
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)  # [bs, 1024, 8, 8]
            f = self.projection(x)  # [bs, 1024, 8, 8]
            x = self.layer4(x)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x1 = self.linear1(x)
            x2 = self.linear2(x)
            return [x1, x2], f


def resNet(model_type="ResNet18", num_classes=10, mode="default", nograd=False):
    model = ResNet(model_type, num_classes, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model
