from models.pose.hourglass import hg as HG
from models.pose.LitePose import litePose as LitePose
from .classification.class_model import class_model as ClassModel
from .pose.pose_model import pose_model as PoseModel


__all__ = ("PoseModel", "ClassModel")
