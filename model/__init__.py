from .vgg import vgg11_bn
from .resnet import resnet50

__factory__ = {
    "vgg": vgg11_bn,
    "resnet": resnet50
}


def get_model(name):
    if name not in __factory__:
        raise ValueError("only support: {}".format(__factory__.keys()))
    return __factory__[name]()
