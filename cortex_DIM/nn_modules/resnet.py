'''Module for making resnet encoders.

'''

import torch
import torch.nn as nn

from cortex_DIM.nn_modules.convnet import Convnet
from cortex_DIM.nn_modules.misc import Fold, Unfold, View


_nonlin_idx = 6


class ResBlock(Convnet):
    '''Residual block for ResNet

    '''

    def create_layers(self, shape, conv_args=None):
        '''Creates layers

        Args:
            shape: Shape of input.
            conv_args: Layer arguments for block.
        '''

        # Move nonlinearity to a separate step for residual.
        final_nonlin = conv_args[-1][_nonlin_idx]
        conv_args[-1] = list(conv_args[-1])
        conv_args[-1][_nonlin_idx] = None
        conv_args.append((None, 0, 0, 0, False, False, final_nonlin, None))

        super().create_layers(shape, conv_args=conv_args)

        if self.conv_shape != shape:
            dim_x, dim_y, dim_in = shape
            dim_x_, dim_y_, dim_out = self.conv_shape
            stride = dim_x // dim_x_
            next_x, _ = self.next_size(dim_x, dim_y, 1, stride, 0)
            assert next_x == dim_x_, (self.conv_shape, shape)

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

        x = self.conv_layers[-1](self.conv_layers[:-1](x) + residual)

        return x


class ResNet(Convnet):
    def create_layers(self, shape, conv_before_args=None, res_args=None, conv_after_args=None, fc_args=None):
        '''Creates layers

        Args:
            shape: Shape of the input.
            conv_before_args: Arguments for convolutional layers before residuals.
            res_args: Residual args.
            conv_after_args: Arguments for convolutional layers after residuals.
            fc_args: Fully-connected arguments.

        '''

        dim_x, dim_y, dim_in = shape
        shape = (dim_x, dim_y, dim_in)
        self.conv_before_layers, self.conv_before_shape = self.create_conv_layers(shape, conv_before_args)
        self.res_layers, self.res_shape = self.create_res_layers(self.conv_before_shape, res_args)
        self.conv_after_layers, self.conv_after_shape = self.create_conv_layers(self.res_shape, conv_after_args)

        dim_x, dim_y, dim_out = self.conv_after_shape
        dim_r = dim_x * dim_y * dim_out
        self.reshape = View(-1, dim_r)
        self.fc_layers, _ = self.create_linear_layers(dim_r, fc_args)

    def create_res_layers(self, shape, block_args=None):
        '''Creates a set of residual blocks.

        Args:
            shape: input shape.
            block_args: Arguments for blocks.

        Returns:
            nn.Sequential: sequence of residual blocks.

        '''

        res_layers = nn.Sequential()
        block_args = block_args or []

        for i, (conv_args, n_blocks) in enumerate(block_args):
            block = ResBlock(shape, conv_args=conv_args)
            res_layers.add_module('block_{}_0'.format(i), block)

            for j in range(1, n_blocks):
                shape = block.conv_shape
                block = ResBlock(shape, conv_args=conv_args)
                res_layers.add_module('block_{}_{}'.format(i, j), block)
            shape = block.conv_shape

        return res_layers, shape

    def forward(self, x: torch.Tensor, return_full_list=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''

        if return_full_list:
            conv_before_out = []
            for conv_layer in self.conv_before_layers:
                x = conv_layer(x)
                conv_before_out.append(x)
        else:
            conv_before_out = self.conv_layers(x)
            x = conv_before_out

        if return_full_list:
            res_out = []
            for res_layer in self.res_layers:
                x = res_layer(x)
                res_out.append(x)
        else:
            res_out = self.res_layers(x)
            x = res_out

        if return_full_list:
            conv_after_out = []
            for conv_layer in self.conv_after_layers:
                x = conv_layer(x)
                conv_after_out.append(x)
        else:
            conv_after_out = self.conv_after_layers(x)
            x = conv_after_out

        x = self.reshape(x)

        if return_full_list:
            fc_out = []
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
                fc_out.append(x)
        else:
            fc_out = self.fc_layers(x)

        return conv_before_out, res_out, conv_after_out, fc_out


class FoldedResNet(ResNet):
    '''Resnet with strided crop input.

    '''

    def create_layers(self, shape, crop_size=8, conv_before_args=None, res_args=None,
                      conv_after_args=None, fc_args=None):
        '''Creates layers

        Args:
            shape: Shape of the input.
            crop_size: Size of the crops.
            conv_before_args: Arguments for convolutional layers before residuals.
            res_args: Residual args.
            conv_after_args: Arguments for convolutional layers after residuals.
            fc_args: Fully-connected arguments.

        '''
        self.crop_size = crop_size

        dim_x, dim_y, dim_in = shape
        self.final_size = 2 * (dim_x // self.crop_size) - 1

        self.unfold = Unfold(dim_x, self.crop_size)
        self.refold = Fold(dim_x, self.crop_size)

        shape = (self.crop_size, self.crop_size, dim_in)
        self.conv_before_layers, self.conv_before_shape = self.create_conv_layers(shape, conv_before_args)

        self.res_layers, self.res_shape = self.create_res_layers(self.conv_before_shape, res_args)
        self.conv_after_layers, self.conv_after_shape = self.create_conv_layers(self.res_shape, conv_after_args)
        self.conv_after_shape = self.res_shape

        dim_x, dim_y, dim_out = self.conv_after_shape
        dim_r = dim_x * dim_y * dim_out
        self.reshape = View(-1, dim_r)
        self.fc_layers, _ = self.create_linear_layers(dim_r, fc_args)

    def create_res_layers(self, shape, block_args=None):
        '''Creates a set of residual blocks.

        Args:
            shape: input shape.
            block_args: Arguments for blocks.

        Returns:
            nn.Sequential: sequence of residual blocks.

        '''

        res_layers = nn.Sequential()
        block_args = block_args or []

        for i, (conv_args, n_blocks) in enumerate(block_args):
            block = ResBlock(shape, conv_args=conv_args)
            res_layers.add_module('block_{}_0'.format(i), block)

            for j in range(1, n_blocks):
                shape = block.conv_shape
                block = ResBlock(shape, conv_args=conv_args)
                res_layers.add_module('block_{}_{}'.format(i, j), block)
            shape = block.conv_shape
            dim_x, dim_y = shape[:2]

            if dim_x != dim_y:
                raise ValueError('dim_x and dim_y do not match.')

            if dim_x == 1:
                shape = (self.final_size, self.final_size, shape[2])

        return res_layers, shape

    def forward(self, x: torch.Tensor, return_full_list=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''
        x = self.unfold(x)

        conv_before_out = []
        for conv_layer in self.conv_before_layers:
            x = conv_layer(x)
            if x.size(2) == 1:
                x = self.refold(x)
            conv_before_out.append(x)

        res_out = []
        for res_layer in self.res_layers:
            x = res_layer(x)
            res_out.append(x)

        if x.size(2) == 1:
            x = self.refold(x)
            res_out[-1] = x

        conv_after_out = []
        for conv_layer in self.conv_after_layers:
            x = conv_layer(x)
            if x.size(2) == 1:
                x = self.refold(x)
            conv_after_out.append(x)

        x = self.reshape(x)

        if return_full_list:
            fc_out = []
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
                fc_out.append(x)
        else:
            fc_out = self.fc_layers(x)

        if not return_full_list:
            conv_before_out = conv_before_out[-1]
            res_out = res_out[-1]
            conv_after_out = conv_after_out[-1]

        return conv_before_out, res_out, conv_after_out, fc_out
