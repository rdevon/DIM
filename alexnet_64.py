'''Alexnet architecture for 64x64 images.

Derived from https://github.com/jeffdonahue/bigan

'''

from cortex.built_ins.networks.modules import View
from cortex.built_ins.networks.base_network import BaseNet
import torch.nn as nn


class AlexNetEncoder(BaseNet):
    def __init__(self, shape, dim_out=None,
                 fully_connected_layers=None,
                 nonlinearity='ReLU',
                 output_nonlinearity=None,
                 **layer_args):
        super(AlexNetEncoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        dim_x, dim_y, dim_in = shape

        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]

        dim_out_ = dim_out
        # AlexNet as in BiGAN paper for 64x64
        self.models = nn.Sequential(
            nn.Conv2d(dim_in, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # weight initilization as in BiGAN paper
        for m in self.models.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.02)
        dim_out = 7 * 7 * 192

        self.models.add_module('final_reshape', View(-1, dim_out))
        dim_out = self.add_linear_layers(dim_out, fully_connected_layers,
                                         **layer_args)
        self.add_output_layer(dim_out, dim_out_)

        for m in self.models.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.02)
