from .MobileNet import MobileNet
from .ResNet import ResNet
from .VGG import VGG


def class_model(modelType, num_classes=10, mode="default", nograd=False):
    if "MobileNet" in modelType:
        model = MobileNet(num_classes, mode).cuda()
    elif "VGG" in modelType:
        model = VGG(modelType, num_classes, mode).cuda()
    elif "ResNet" in modelType:
        model = ResNet(modelType, num_classes, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model