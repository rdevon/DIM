'''Convnet encoder module.

'''

import torch
import torch.nn as nn

from cortex.built_ins.networks.utils import get_nonlinearity

from cortex_DIM.nn_modules.misc import Fold, Unfold, View


def infer_conv_size(w, k, s, p):
    '''Infers the next size after convolution.

    Args:
        w: Input size.
        k: Kernel size.
        s: Stride.
        p: Padding.

    Returns:
        int: Output size.

    '''
    x = (w - k + 2 * p) // s + 1
    return x


class Convnet(nn.Module):
    '''Basic convnet convenience class.

    Attributes:
        conv_layers: nn.Sequential of nn.Conv2d layers with batch norm,
            dropout, nonlinearity.
        fc_layers: nn.Sequential of nn.Linear layers with batch norm,
            dropout, nonlinearity.
        reshape: Simple reshape layer.
        conv_shape: Shape of the convolutional output.

    '''

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.create_layers(*args, **kwargs)

    def create_layers(self, shape, conv_args=None, fc_args=None):
        '''Creates layers

        conv_args are in format (dim_h, f_size, stride, pad, batch_norm, dropout, nonlinearity, pool)
        fc_args are in format (dim_h, batch_norm, dropout, nonlinearity)

        Args:
            shape: Shape of input.
            conv_args: List of tuple of convolutional arguments.
            fc_args: List of tuple of fully-connected arguments.
        '''

        if len(shape) == 1:
            shape = (1, 1, shape[0])

        self.conv_layers, self.conv_shape = self.create_conv_layers(shape, conv_args)

        dim_x, dim_y, dim_out = self.conv_shape
        dim_r = dim_x * dim_y * dim_out
        self.reshape = View(-1, dim_r)
        self.fc_layers, _ = self.create_linear_layers(dim_r, fc_args)

    def create_conv_layers(self, shape, conv_args):
        '''Creates a set of convolutional layers.

        Args:
            shape: Input shape.
            conv_args: List of tuple of convolutional arguments.

        Returns:
            nn.Sequential: a sequence of convolutional layers.

        '''

        conv_layers = nn.Sequential()
        conv_args = conv_args or []

        dim_x, dim_y, dim_in = shape

        for i, (dim_out, f, s, p, batch_norm, dropout, nonlinearity, pool) in enumerate(conv_args):
            name = '({}/{})_{}'.format(dim_in, dim_out, i + 1)
            conv_block = nn.Sequential()

            if dim_out is not None:
                conv = nn.Conv2d(dim_in, dim_out, kernel_size=f, stride=s, padding=p, bias=not(batch_norm))
                conv_block.add_module(name + 'conv', conv)
                dim_x, dim_y = self.next_size(dim_x, dim_y, f, s, p)
            else:
                dim_out = dim_in

            if dropout:
                conv_block.add_module(name + 'do', nn.Dropout2d(p=dropout))
            if batch_norm:
                bn = nn.BatchNorm2d(dim_out)
                conv_block.add_module(name + 'bn', bn)

            if nonlinearity:
                nonlinearity = get_nonlinearity(nonlinearity)
                conv_block.add_module(nonlinearity.__class__.__name__, nonlinearity)

            if pool:
                (pool_type, kernel, stride) = pool
                Pool = getattr(nn, pool_type)
                conv_block.add_module(name + 'pool', Pool(kernel_size=kernel, stride=stride))
                dim_x, dim_y = self.next_size(dim_x, dim_y, kernel, stride, 0)

            conv_layers.add_module(name, conv_block)

            dim_in = dim_out

        dim_out = dim_in

        return conv_layers, (dim_x, dim_y, dim_out)

    def create_linear_layers(self, dim_in, fc_args):
        '''

        Args:
            dim_in: Number of input units.
            fc_args: List of tuple of fully-connected arguments.

        Returns:
            nn.Sequential.

        '''

        fc_layers = nn.Sequential()
        fc_args = fc_args or []

        for i, (dim_out, batch_norm, dropout, nonlinearity) in enumerate(fc_args):
            name = '({}/{})_{}'.format(dim_in, dim_out, i + 1)
            fc_block = nn.Sequential()

            if dim_out is not None:
                fc_block.add_module(name + 'fc', nn.Linear(dim_in, dim_out))
            else:
                dim_out = dim_in

            if dropout:
                fc_block.add_module(name + 'do', nn.Dropout(p=dropout))
            if batch_norm:
                bn = nn.BatchNorm1d(dim_out)
                fc_block.add_module(name + 'bn', bn)

            if nonlinearity:
                nonlinearity = get_nonlinearity(nonlinearity)
                fc_block.add_module(nonlinearity.__class__.__name__, nonlinearity)

            fc_layers.add_module(name, fc_block)

            dim_in = dim_out

        return fc_layers, dim_in

    def next_size(self, dim_x, dim_y, k, s, p):
        '''Infers the next size of a convolutional layer.

        Args:
            dim_x: First dimension.
            dim_y: Second dimension.
            k: Kernel size.
            s: Stride.
            p: Padding.

        Returns:
            (int, int): (First output dimension, Second output dimension)

        '''
        if isinstance(k, int):
            kx, ky = (k, k)
        else:
            kx, ky = k

        if isinstance(s, int):
            sx, sy = (s, s)
        else:
            sx, sy = s

        if isinstance(p, int):
            px, py = (p, p)
        else:
            px, py = p
        return (infer_conv_size(dim_x, kx, sx, px),
                infer_conv_size(dim_y, ky, sy, py))

    def forward(self, x: torch.Tensor, return_full_list=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''
        if return_full_list:
            conv_out = []
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
                conv_out.append(x)
        else:
            conv_out = self.conv_layers(x)
            x = conv_out

        x = self.reshape(x)

        if return_full_list:
            fc_out = []
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
                fc_out.append(x)
        else:
            fc_out = self.fc_layers(x)

        return conv_out, fc_out


class FoldedConvnet(Convnet):
    '''Convnet with strided crop input.

    '''

    def create_layers(self, shape, crop_size=8, conv_args=None, fc_args=None):
        '''Creates layers

        conv_args are in format (dim_h, f_size, stride, pad, batch_norm, dropout, nonlinearity, pool)
        fc_args are in format (dim_h, batch_norm, dropout, nonlinearity)

        Args:
            shape: Shape of input.
            crop_size: Size of crops
            conv_args: List of tuple of convolutional arguments.
            fc_args: List of tuple of fully-connected arguments.
        '''

        self.crop_size = crop_size

        dim_x, dim_y, dim_in = shape
        if dim_x != dim_y:
            raise ValueError('x and y dimensions must be the same to use Folded encoders.')

        self.final_size = 2 * (dim_x // self.crop_size) - 1

        self.unfold = Unfold(dim_x, self.crop_size)
        self.refold = Fold(dim_x, self.crop_size)

        shape = (self.crop_size, self.crop_size, dim_in)

        self.conv_layers, self.conv_shape = self.create_conv_layers(shape, conv_args)

        dim_x, dim_y, dim_out = self.conv_shape
        dim_r = dim_x * dim_y * dim_out
        self.reshape = View(-1, dim_r)
        self.fc_layers, _ = self.create_linear_layers(dim_r, fc_args)

    def create_conv_layers(self, shape, conv_args):
        '''Creates a set of convolutional layers.

        Args:
            shape: Input shape.
            conv_args: List of tuple of convolutional arguments.

        Returns:
            nn.Sequential: A sequence of convolutional layers.

        '''

        conv_layers = nn.Sequential()
        conv_args = conv_args or []
        dim_x, dim_y, dim_in = shape

        for i, (dim_out, f, s, p, batch_norm, dropout, nonlinearity, pool) in enumerate(conv_args):
            name = '({}/{})_{}'.format(dim_in, dim_out, i + 1)
            conv_block = nn.Sequential()

            if dim_out is not None:
                conv = nn.Conv2d(dim_in, dim_out, kernel_size=f, stride=s, padding=p, bias=not(batch_norm))
                conv_block.add_module(name + 'conv', conv)
                dim_x, dim_y = self.next_size(dim_x, dim_y, f, s, p)
            else:
                dim_out = dim_in

            if dropout:
                conv_block.add_module(name + 'do', nn.Dropout2d(p=dropout))
            if batch_norm:
                bn = nn.BatchNorm2d(dim_out)
                conv_block.add_module(name + 'bn', bn)

            if nonlinearity:
                nonlinearity = get_nonlinearity(nonlinearity)
                conv_block.add_module(nonlinearity.__class__.__name__, nonlinearity)

            if pool:
                (pool_type, kernel, stride) = pool
                Pool = getattr(nn, pool_type)
                conv_block.add_module('pool', Pool(kernel_size=kernel, stride=stride))
                dim_x, dim_y = self.next_size(dim_x, dim_y, kernel, stride, 0)

            conv_layers.add_module(name, conv_block)

            dim_in = dim_out

            if dim_x != dim_y:
                raise ValueError('dim_x and dim_y do not match.')

            if dim_x == 1:
                dim_x = self.final_size
                dim_y = self.final_size

        dim_out = dim_in

        return conv_layers, (dim_x, dim_y, dim_out)

    def forward(self, x: torch.Tensor, return_full_list=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''

        x = self.unfold(x)

        conv_out = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if x.size(2) == 1:
                x = self.refold(x)
            conv_out.append(x)

        x = self.reshape(x)

        if return_full_list:
            fc_out = []
            for fc_layer in self.fc_layers:
                x = fc_layer(x)
                fc_out.append(x)
        else:
            fc_out = self.fc_layers(x)

        if not return_full_list:
            conv_out = conv_out[-1]

        return conv_out, fc_out