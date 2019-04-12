'''Module for prior matching.

'''

import torch.nn as nn

from cortex_DIM.functions.gan_losses import generator_loss
from cortex_DIM.models.discriminator import Discriminator
from cortex_DIM.nn_modules.convnet import Convnet


class PriorMatching(Discriminator):
    '''Model for matching the output of a network to a prior.

    '''

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='data.images'), skip_last_batch=True),
        train=dict(save_on_lowest='losses.encoder', epochs=1000),
        optimizer=dict(learning_rate=1e-4)
    )

    def build(self, encoder, config,
              discriminator_layers=[dict(layer='linear', args=(1000,), act='ReLU'),
                                    dict(layer='linear', args=(200,), act='ReLU')],
              task_idx=-1, dist='uniform', use_nonlinearity=True):
        '''

        Args:
            discriminator_layers: Layers for the discriminator.
            task_idx: Index from the encoder to apply prior matching.
            dist: Distribution of the prior.
            use_nonlinearity: Use nonlinearity for uniform distribution.

        '''
        self.nets.encoder = encoder

        self.task_idx = task_idx

        X = self.inputs('data.images')
        Z = self.nets.encoder(X, return_all_activations=True)[self.task_idx]

        nonlinearity = None
        if dist == 'uniform' and use_nonlinearity:
            nonlinearity = nn.Sigmoid()
        self.nonlinearity = nonlinearity

        self.add_noise('prior', dist=dist, size=Z.size()[1:])
        super().build(Z.size()[1:], discriminator_args=dict(Module=Convnet, layers=discriminator_layers))

    def routine(self, outs=None, measure='GAN', scale=1.0):
        '''

        Args:
            measure: Probability measure to use for prior matching.
            scale: Hyperparameter for prior matching. Called `gamma` in the paper.

        '''
        if outs is None:
            X = self.inputs('data.images')
            outs = self.nets.encoder(X, return_all_activations=True)

        Z_P = self.inputs('prior')
        Z_Q = outs[self.task_idx]

        if self.nonlinearity is not None:
            Z_Q = self.nonlinearity(Z_Q)

        super().routine(Z_P, Z_Q.detach(), measure=measure)

        if scale > 0:
            Q_samples = self.score(Z_P, Z_Q, measure)[3]
            prior_loss = generator_loss(Q_samples, measure, 'non-saturating')
            self.add_losses(encoder=scale * prior_loss)

    def visualize(self, measure=None):
        X, Z_P = self.inputs('data.images', 'prior')
        Z_Q = self.nets.encoder(X, return_all_activations=True)[self.task_idx]

        if self.nonlinearity is not None:
            Z_Q = self.nonlinearity(Z_Q)

        self.add_histogram(
            dict(real=Z_P.view(-1).data, fake=Z_Q.view(-1).data),
            name='encoder_output')

        super().visualize(Z_P, Z_Q, measure=measure)
