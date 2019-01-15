'''Basic convnet hyperparameters.

conv_args are in format (dim_h, f_size, stride, pad batch_norm, dropout, nonlinearity, pool)
fc_args are in format (dim_h, batch_norm, dropout, nonlinearity)

'''

from cortex_DIM.nn_modules.encoder import ConvnetEncoder, FoldedConvnetEncoder


# Basic DCGAN-like encoders

_basic28x28 = dict(
    Encoder=ConvnetEncoder,
    conv_args=[(64, 5, 2, 2, True, False, 'ReLU', None),
               (128, 5, 2, 2, True, False, 'ReLU', None)],
    fc_args=[(1024, True, False, 'ReLU', None)],
    local_idx=1,
    fc_idx=0
)

_basic32x32 = dict(
    Encoder=ConvnetEncoder,
    conv_args=[(64, 4, 2, 1, True, False, 'ReLU', None),
               (128, 4, 2, 1, True, False, 'ReLU', None),
               (256, 4, 2, 1, True, False, 'ReLU', None)],
    fc_args=[(1024, True, False, 'ReLU')],
    local_idx=1,
    conv_idx=2,
    fc_idx=0
)

_basic64x64 = dict(
    Encoder=ConvnetEncoder,
    conv_args=[(64, 4, 2, 1, True, False, 'ReLU', None),
               (128, 4, 2, 1, True, False, 'ReLU', None),
               (256, 4, 2, 1, True, False, 'ReLU', None),
               (512, 4, 2, 1, True, False, 'ReLU', None)],
    fc_args=[(1024, True, False, 'ReLU')],
    local_idx=2,
    conv_idx=3,
    fc_idx=0
)

# Alexnet-like encoders

_alex64x64 = dict(
    Encoder=ConvnetEncoder,
    conv_args=[(96, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (192, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (384, 3, 1, 1, True, False, 'ReLU', None),
               (384, 3, 1, 1, True, False, 'ReLU', None),
               (192, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2))],
    fc_args=[(4096, True, False, 'ReLU'),
             (4096, True, False, 'ReLU')],
    local_idx=2,
    conv_idx=4,
    fc_idx=1
)

_foldalex64x64 = dict(
    Encoder=FoldedConvnetEncoder,
    crop_size=16,
    conv_args=[(96, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (192, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (384, 3, 1, 1, True, False, 'ReLU', None),
               (384, 3, 1, 1, True, False, 'ReLU', None),
               (192, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2))],
    fc_args=[(4096, True, False, 'ReLU'),
             (4096, True, False, 'ReLU')],
    local_idx=4,
    fc_idx=1
)

_foldmultialex64x64 = dict(
    Encoder=FoldedConvnetEncoder,
    crop_size=16,
    conv_args=[(96, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (192, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (384, 3, 1, 1, True, False, 'ReLU', None),
               (384, 3, 1, 1, True, False, 'ReLU', None),
               (192, 3, 1, 1, True, False, 'ReLU', ('MaxPool2d', 3, 2)),
               (192, 3, 1, 0, True, False, 'ReLU', None),
               (192, 1, 1, 0, True, False, 'ReLU', None)],
    fc_args=[(4096, True, False, 'ReLU')],
    local_idx=4,
    multi_idx=6,
    fc_idx=1
)

configs = dict(
    basic28x28=_basic28x28,
    basic32x32=_basic32x32,
    basic64x64=_basic64x64,
    alex64x64=_alex64x64,
    foldalex64x64=_foldalex64x64,
    foldmultialex64x64=_foldmultialex64x64
)