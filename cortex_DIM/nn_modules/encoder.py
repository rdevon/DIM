'''Basic cortex_DIM encoder.

'''

import torch

from cortex_DIM.nn_modules.convnet import Convnet, FoldedConvnet
from cortex_DIM.nn_modules.resnet import ResNet, FoldedResNet


def create_encoder(Module):
    class Encoder(Module):
        '''Encoder used for cortex_DIM.

        '''

        def __init__(self, *args, local_idx=None, multi_idx=None, conv_idx=None, fc_idx=None, **kwargs):
            '''

            Args:
                args: Arguments for parent class.
                local_idx: Index in list of convolutional layers for local features.
                multi_idx: Index in list of convolutional layers for multiple globals.
                conv_idx: Index in list of convolutional layers for intermediate features.
                fc_idx: Index in list of fully-connected layers for intermediate features.
                kwargs: Keyword arguments for the parent class.
            '''

            super().__init__(*args, **kwargs)

            if local_idx is None:
                raise ValueError('`local_idx` must be set')

            conv_idx = conv_idx or local_idx

            self.local_idx = local_idx
            self.multi_idx = multi_idx
            self.conv_idx = conv_idx
            self.fc_idx = fc_idx

        def forward(self, x: torch.Tensor):
            '''

            Args:
                x: Input tensor.

            Returns:
                local_out, multi_out, hidden_out, global_out

            '''

            outs = super().forward(x, return_full_list=True)
            if len(outs) == 2:
                conv_out, fc_out = outs
            else:
                conv_before_out, res_out, conv_after_out, fc_out = outs
                conv_out = conv_before_out + res_out + conv_after_out

            local_out = conv_out[self.local_idx]

            if self.multi_idx is not None:
                multi_out = conv_out[self.multi_idx]
            else:
                multi_out = None

            if len(fc_out) > 0:
                if self.fc_idx is not None:
                    hidden_out = fc_out[self.fc_idx]
                else:
                    hidden_out = None
                global_out = fc_out[-1]
            else:
                hidden_out = None
                global_out = None

            conv_out = conv_out[self.conv_idx]

            return local_out, conv_out, multi_out, hidden_out, global_out

    return Encoder


class ConvnetEncoder(create_encoder(Convnet)):
    pass


class FoldedConvnetEncoder(create_encoder(FoldedConvnet)):
    pass


class ResnetEncoder(create_encoder(ResNet)):
    pass


class FoldedResnetEncoder(create_encoder(FoldedResNet)):
    pass
