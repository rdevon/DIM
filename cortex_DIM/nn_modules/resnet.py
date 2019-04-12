'''Module for making resnet encoders.

'''

import torch
import torch.nn as nn

from cortex_DIM.nn_modules.convnet import Convnet
from cortex_DIM.nn_modules.misc import Fold, Unfold


class ResBlock(Convnet):
    '''Residual block for ResNet

    '''

    def create_layers(self, shape, layers=None, final_act=None, downsample=None):
        '''Creates layers

        Args:
            shape: Shape of input.
            layers: list of layer arguments.
            final_act: Final activation.
            downsample: Arguments for downsample (optional).
        '''

        # Move nonlinearity to a separate step for residual.
        self.layers, self.shapes = self.create_sequential(shape, layers=layers)
        self.final_act, _ = self.create_sequential(self.shapes[-1], layers=[dict(act=final_act)])

        if self.shapes[-1] != shape:
            if downsample is not None:
                self.downsample, ds_shape = self.create_sequential(shape, layers=downsample)
                assert ds_shape[-1] == self.shapes[-1], (ds_shape[-1], self.shapes[-1])
            else:
                # Auto build downsample (doesn't always work)
                dim_x, dim_y, dim_in = shape
                dim_x_, dim_y_, dim_out = self.shapes[-1]
                stride = dim_x // dim_x_
                next_x, _ = self.next_conv_size(dim_x, dim_y, 1, stride, 0)
                assert next_x == dim_x_, (self.shapes[-1], shape)

                self.downsample = nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(dim_out),
                )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        '''Forward pass

        Args:
            x: Input.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.final_act(self.layers(x) + residual)

        return x


class ResNet(Convnet):
    '''Resnet.

    '''

    _supported_types = ('linear', 'conv', 'tconv', 'flatten', 'resblock', None)

    def handle_layer(self, block, shape, layer, layer_type):
        '''Handles the layer arguments and adds layer to the block.

        Args:
            block: nn.Sequential to add modules to.
            shape: Shape of the input.
            layer: Layer arguments.
            layer_type: Type of layer.

        Returns:
            tuple: Output shape.
        '''

        if layer_type == 'resblock':
            layers = layer.pop('layers', [])
            final_act = layer.pop('final_act', None)
            downsample = layer.pop('downsample', None)
            repeat = layer.pop('repeat', 1)
            if repeat == 1:
                resblock = ResBlock(shape, layers=layers, final_act=final_act,
                                    downsample=downsample)
                block.add_module(layer_type, resblock)
                shape = resblock.shapes[-1]
            else:
                for i in range(repeat):
                    resblock = ResBlock(shape, layers=layers, final_act=final_act,
                                        downsample=downsample)
                    block.add_module(layer_type + '_{}'.format(i), resblock)
                    shape = resblock.shapes[-1]
        else:
            shape = super().handle_layer(block, shape, layer, layer_type)

        return shape


class FoldedResNet(ResNet):
    '''Resnet with strided crop input.

    '''

    _supported_types = ('linear', 'conv', 'tconv', 'flatten', 'resblock', 'fold', 'unfold', None)

    def create_layers(self, shape, crop_size=8, layers=None):
        ''''Creates layers

        Args:
            shape: Shape of input.
            crop_size: Size of crops
            layers: list of layer arguments.
        '''

        self.crop_size = crop_size
        self.layers, self.shapes = self.create_sequential(shape, layers=layers)

    def create_sequential(self, shape, layers=None):
        '''Creates a sequence of layers.

        Args:
            shape: Input shape.
            layers: list of layer arguments.

        Returns:
            nn.Sequential: a sequence of convolutional layers.

        '''

        self.final_size = None
        return super().create_sequential(shape, layers=layers)

    def handle_layer(self, block, shape, layer, layer_type):
        '''Handles the layer arguments and adds layer to the block.

        Args:
            block: nn.Sequential to add modules to.
            shape: Shape of the input.
            layer: Layer arguments.
            layer_type: Type of layer.

        Returns:
            tuple: Output shape.
        '''
        if layer_type == 'unfold':
            dim_x, dim_y, dim_out = shape
            self.final_size = 2 * (dim_x // self.crop_size) - 1
            block.add_module('unfold', Unfold(dim_x, self.crop_size))
            shape = (self.crop_size, self.crop_size, dim_out)
        elif layer_type == 'fold':
            if self.final_size is None:
                raise ValueError('Cannot fold without unfolding first.')
            dim_out = shape[2]
            block.add_module('fold', Fold(self.final_size))
            shape = (self.final_size, self.final_size, dim_out)
        elif layer_type is None:
            pass
        else:
            shape = super().handle_layer(block, shape, layer, layer_type)

        return shape
