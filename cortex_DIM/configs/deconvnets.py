'''Decoder configurations.

See `configs.convnets` for details.

'''

from cortex_DIM.nn_modules.convnet import Convnet


_basic32x32 = dict(
    Module=Convnet,
    layers=[dict(layer='linear', args=(512 * 4 * 4,), bn=True, act='ReLU'),
            dict(layer='reshape', args=(512, 4, 4)),
            dict(layer='tconv', args=(256, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='tconv', args=(128, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='tconv', args=(64, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(3, 3, 1, 1), bias=False, act='Tanh')]
)

configs = dict(
    basic32x32=_basic32x32
)