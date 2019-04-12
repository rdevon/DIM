'''Basic decoder model trained on reconstruction.

'''

import torch.nn.functional as F
from cortex.plugins import ModelPlugin

from cortex_DIM.functions.misc import ms_ssim


class Decoder(ModelPlugin):
    '''Basic decoder model trained on reconstruction.

    '''

    def build(self, shape, args=dict()):
        '''

        Args:
            shape: Input shape.
            args (dict): Arguments for the decoder.

        '''

        Module = args['Module']
        layers = args['layers']
        decoder = Module(shape, layers=layers)

        self.add_nets(decoder=decoder)

    def routine(self, inputs, Z, decoder_crit=F.mse_loss):
        '''

        Args:
            decoder_crit: Criteria for reconstruction objective.

        '''
        X = self.nets.decoder(Z)
        msssim = ms_ssim(inputs, X)
        self.add_losses(decoder=decoder_crit(X, inputs))
        self.add_results(MS_SSIM=msssim.detach().item())

    def visualize(self, Z):
        gen = self.nets.decoder(Z)
        self.add_image(gen, name='Decoder generated')
