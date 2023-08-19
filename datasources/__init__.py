from .lsp import LSPData as LSP
from .mouse import MouseData as Mouse
from .flic import FLICData as FLIC
from .fly import FLYData as Fly
from .pranav import PranavData as Pranav
from .ap10k import AP10KData as AP10K
from .classification.cifar10 import CIFAR10Data as CIFAR10
from .classification.cifar100 import CIFAR100Data as CIFAR100

__all__ = ("Mouse", "Pranav", "Fly", "AP10K", "LSP", "FLIC", "CIFAR10", "CIFAR100")