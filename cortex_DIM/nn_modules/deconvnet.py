'''Reverse convnet.

'''

import torch.nn as nn

from cortex_DIM.nn_modules.convnet import Convnet


class DeConvnet(Convnet):
    '''Convnet with strided crop input.

    '''

    _supported_types = ('linear', 'conv', 'flatten', None)

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
            block.add_module(layer_type, Unfold(dim_x, self.crop_size))
            shape = (self.crop_size, self.crop_size, dim_out)
        elif layer_type == 'fold':
            if self.final_size is None:
                raise ValueError('Cannot fold without unfolding first.')
            dim_out = shape[2]
            block.add_module(layer_type, Fold(self.final_size))
            shape = (self.final_size, self.final_size, dim_out)
        elif layer_type is None:
            pass
        else:
            shape = super().handle_layer(block, shape, layer, layer_type)

        return shape