"""Configurations for ResNets

"""

from cortex_DIM.nn_modules.encoder import ResnetEncoder, FoldedResnetEncoder


_resnet19_32x32 = dict(
    Encoder=ResnetEncoder,
    conv_before_args=[(64, 3, 2, 1, True, False, 'ReLU', None)],
    res_args=[
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 2, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 1, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 2, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 1, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1)
    ],
    fc_args=[(1024, True, False, 'ReLU')],
    local_idx=4,
    fc_idx=0
)

_foldresnet19_32x32 = dict(
    Encoder=FoldedResnetEncoder,
    crop_size=8,
    conv_before_args=[(64, 3, 2, 1, True, False, 'ReLU', None)],
    res_args=[
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 2, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 1, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 2, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 1, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1)
    ],
    fc_args=[(1024, True, False, 'ReLU')],
    local_idx=6,
    fc_idx=0
)

_resnet34_32x32 = dict(
    Encoder=ResnetEncoder,
    conv_before_args=[(64, 3, 2, 1, True, False, 'ReLU', None)],
    res_args=[
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         2),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 2, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 1, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         5),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 2, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 1, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         2)
    ],
    fc_args=[(1024, True, False, 'ReLU')],
    local_idx=2,
    fc_idx=0
)

_foldresnet34_32x32 = dict(
    Encoder=FoldedResnetEncoder,
    crop_size=8,
    conv_before_args=[(64, 3, 2, 1, True, False, 'ReLU', None)],
    res_args=[
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(64, 1, 1, 0, True, False, 'ReLU', None),
          (64, 3, 1, 1, True, False, 'ReLU', None),
          (64 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         2),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 2, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(128, 1, 1, 0, True, False, 'ReLU', None),
          (128, 3, 1, 1, True, False, 'ReLU', None),
          (128 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         5),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 2, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         1),
        ([(256, 1, 1, 0, True, False, 'ReLU', None),
          (256, 3, 1, 1, True, False, 'ReLU', None),
          (256 * 4, 1, 1, 0, True, False, 'ReLU', None)],
         2)
    ],
    fc_args=[(1024, True, False, 'ReLU')],
    local_idx=12,
    fc_idx=0
)

configs = dict(
    resnet19_32x32=_resnet19_32x32,
    resnet34_32x32=_resnet34_32x32,
    foldresnet19_32x32=_foldresnet19_32x32,
    foldresnet34_32x32=_foldresnet34_32x32
)