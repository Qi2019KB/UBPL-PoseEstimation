from torch import nn
from models.base.layers import Conv


class MobileNet(nn.Module):
    def __init__(self, num_classes, mode):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.layer1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 96, 2),
            conv_dw(96, 96, 1),
            conv_dw(96, 128, 2),
            conv_dw(128, 128, 1),
        )

        self.layer2 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),
            nn.AvgPool2d(2),
        )

        self.projection = self._get_projection(self.mode)

        self.fc1 = nn.Linear(512, self.num_classes)

        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        if self.mode == "default":
            x = self.layer1(x)       # [bs, 3, 32, 32] ==> [bs, 128, 4, 4]
            x = self.layer2(x)       # [30, 512, 1, 1]
            x = x.view(-1, 512)      # [30, 512]
            x1 = self.fc1(x)           # [30, 10]
            x2 = self.fc2(x)           # [30, 10]
            return x1, x2
        else:
            x = self.layer1(x)       # [bs, 3, 32, 32] ==> [bs, 128, 4, 4]
            f = self.projection(x)  # [bs, 128, 2, 2]
            x = self.layer2(x)       # [30, 512, 1, 1]
            x = x.view(-1, 512)      # [30, 512]
            x1 = self.fc1(x)           # [30, 10]
            x2 = self.fc2(x)           # [30, 10]
            return [x1, x2], f

    def _get_projection(self, mode):
        if mode == "MaxPool":
            return nn.Sequential(nn.MaxPool2d(2, 2))
        elif mode == "AvgPool":
            return nn.Sequential(nn.AvgPool2d(2, 2))
        elif self.mode == "ConvOne":
            return nn.Sequential(nn.Conv2d(128, 128, 1, 1, 0, bias=False))


def mobileNet(num_classes=10, mode="default", nograd=False):
    model = MobileNet(num_classes, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model