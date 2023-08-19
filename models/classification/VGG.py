import  torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, model_type, num_classes, mode):
        super(VGG, self).__init__()
        self.cfg = {
            "VGG11": [[64, "M", 128, "M", 256, 256, "M", 512, 512], ["M", 512, 512, "M"]],
            "VGG13": [[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512], ['M', 512, 512, 'M']],
            'VGG16': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512], ['M', 512, 512, 512, 'M']],
            'VGG19': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512], ['M', 512, 512, 512, 512, 'M']]
        }
        self.model_type = model_type
        self.num_classes = num_classes
        self.mode = mode
        self.layer1 = self._make_layers(self.cfg[self.model_type][0])
        self.layer2 = self._make_layers(self.cfg[self.model_type][1], isNew=False)
        self.projection = self._get_projection(self.mode)
        self.fc1 = nn.Linear(512, self.num_classes)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        if self.mode == "default":
            x = self.layer1(x)       # [bs, 3, 32, 32] ==> [bs, 512, 4, 4]
            x = self.layer2(x)       # [30, 512, 1, 1]
            x = x.view(x.size(0), -1)
            x1 = self.fc1(x)           # [30, 10]
            x2 = self.fc2(x)           # [30, 10]
            return x1, x2
        else:
            x = self.layer1(x)       # [bs, 3, 32, 32] ==> [bs, 512, 4, 4]
            f = self.projection(x)  # [bs, 512, 2, 2]
            x = self.layer2(x)       # [30, 512, 1, 1]
            x = x.view(x.size(0), -1)
            x1 = self.fc1(x)           # [30, 10]
            x2 = self.fc2(x)           # [30, 10]
            return [x1, x2], f

    def _get_projection(self, mode):
        if mode == "MaxPool":
            return nn.Sequential(nn.MaxPool2d(2, 2))
        elif mode == "AvgPool":
            return nn.Sequential(nn.AvgPool2d(2, 2))
        elif self.mode == "ConvOne":
            return nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0, bias=False))

    def _make_layers(self, cfg, isNew=True):
        layers = []
        in_channels = 3

        for layer_para in cfg:
            if layer_para != "M":
                if isNew == False: in_channels = layer_para
                layers += [
                    nn.Conv2d(in_channels, layer_para, kernel_size=3, padding=1),
                    nn.BatchNorm2d(layer_para),
                    nn.ReLU(inplace=True)
                ]
                in_channels = layer_para
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


def vggNet(model_type="VGG11", num_classes=10, mode="default", nograd=False):
    model = VGG(model_type, num_classes, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model
