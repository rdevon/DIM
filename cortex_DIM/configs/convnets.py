'''Basic convnet hyperparameters.

Keywords for each layer include: layer, args, bn (batch norm), do (dropout), act (activation), pool.
conv layer args are (n_units, filter size, stride, pool)
linear layer args are (n_units,)
See cortex_DIM.nn_modules.convnet.py for details.

local_task_idx is a pair of indices (local features layer index, global feature(s) layer index)
classifier_idx is a dictionary with index values for layers on which to build a classifier.

Note that in DIM, a final layer is added for the global variable, so -1 specifies this layer.

'''

from cortex_DIM.nn_modules.convnet import Convnet, FoldedConvnet


# This network is for MNIST.
_basic28x28 = dict(
    Module=Convnet,
    layers=[dict(layer='conv', args=(64, 5, 2, 2), bn=True, act='ReLU'),
            dict(layer='conv', args=(128, 5, 2, 2), bn=True, act='ReLU'),
            dict(layer='flatten'),
            dict(layer='linear', args=(1024,), bn=True, act='ReLU')],
    local_task_idx=(1, -1),
    classifier_idx=dict(conv=1, fc=3, glob=-1)
)

# This network was used in Table 1 CIFAR10 and CIFAR100 results.
_basic32x32 = dict(
    Module=Convnet,
    layers=[dict(layer='conv', args=(64, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(128, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(256, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='flatten'),
            dict(layer='linear', args=(1024,), bn=True, act='ReLU')],
    local_task_idx=(1, -1),
    classifier_idx=dict(conv=2, fc=4, glob=-1)
)

_basic64x64 = dict(
    Module=Convnet,
    layers=[dict(layer='conv', args=(64, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(128, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(256, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(512, 4, 2, 1), bn=True, act='ReLU'),
            dict(layer='flatten'),
            dict(layer='linear', args=(1024,), bn=True, act='ReLU')],
    local_task_idx=(2, -1),
    classifier_idx=dict(conv=3, fc=5, glob=-1)
)

# This network was used for STL10 and tiny Imagenet results.
_alex64x64 = dict(
    Module=Convnet,
    layers=[dict(layer='conv', args=(96, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2)),
            dict(layer='conv', args=(192, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2)),
            dict(layer='conv', args=(384, 3, 1, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(384, 3, 1, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(192, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2)),
            dict(layer='flatten'),
            dict(layer='linear', args=(4096,), bn=True, act='ReLU'),
            dict(layer='linear', args=(4096,), bn=True, act='ReLU')],
    local_task_idx=(2, -1),
    classifier_idx=dict(conv=4, fc=7, glob=-1)
)

# The following two networks were used for STL10 CPC comparison results.
_foldalex64x64 = dict(
    Module=FoldedConvnet,
    crop_size=16,
    layers=[dict(layer='unfold'),
            dict(layer='conv', args=(96, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2),
                 init='normal_',  init_args=dict(mean=0, std=0.02), bias=True),
            dict(layer='conv', args=(192, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2),
                 init='normal_', init_args=dict(mean=0, std=0.02), bias=True),
            dict(layer='conv', args=(384, 3, 1, 1), bn=True, act='ReLU', init='normal_',
                 init_args=dict(mean=0, std=0.02), bias=True),
            dict(layer='conv', args=(384, 3, 1, 1), bn=True, act='ReLU', init='normal_',
                 init_args=dict(mean=0, std=0.02), bias=True),
            dict(layer='conv', args=(192, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2),
                 init='normal_', init_args=dict(mean=0, std=0.02), bias=True),
            dict(layer='fold'),
            dict(layer='conv', args=(192*2, 3, 2, 0), do=0.1, bn=True, act='ReLU', bias=False),
            dict(layer='linear', args=(4096,), bn=True, act='ReLU', init='normal_',
                 init_args=dict(mean=0, std=0.02), bias=True, do=0.1)
            ],
    local_task_idx=(6, -1),
    classifier_idx=dict(conv=6, fc=8, glob=-1)
)

_foldmultialex64x64 = dict(
    Module=FoldedConvnet,
    crop_size=16,
    layers=[dict(layer='unfold'),
            dict(layer='conv', args=(96, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2)),
            dict(layer='conv', args=(192, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2)),
            dict(layer='conv', args=(384, 3, 1, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(384, 3, 1, 1), bn=True, act='ReLU'),
            dict(layer='conv', args=(192, 3, 1, 1), bn=True, act='ReLU', pool=('MaxPool2d', 3, 2)),
            dict(layer='fold'),
            dict(layer='conv', args=(192, 3, 1, 0), bn=True, act='ReLU'),
            dict(layer='conv', args=(192, 1, 1, 0), bn=True, act='ReLU')],
    local_task_idx=(6, 8),
    classifier_idx=dict(conv=6, glob=8)
)

configs = dict(
    basic28x28=_basic28x28,
    basic32x32=_basic32x32,
    basic64x64=_basic64x64,
    alex64x64=_alex64x64,
    foldalex64x64=_foldalex64x64,
    foldmultialex64x64=_foldmultialex64x64
)