from .hourglass import StackedHourglass
from .LitePose import LitePose


def pose_model(modelType, kpsCount, mode="default", nograd=False):
    if "HG" in modelType:
        nStack = int(modelType[len("HG"):])
        model = StackedHourglass(kpsCount, nStack, mode).cuda()
    elif "LitePose" in modelType:
        model = LitePose(kpsCount, mode).cuda()
    if nograd:
        for param in model.parameters():
            param.detach_()
    return model