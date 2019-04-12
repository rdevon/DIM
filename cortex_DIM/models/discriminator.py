'''Basic GAN discriminator

'''

import copy

from cortex.plugins import ModelPlugin

from cortex_DIM.functions.gradient_penalty import contrastive_gradient_penalty
from cortex_DIM.functions.gan_losses import get_negative_expectation, get_positive_expectation


class Discriminator(ModelPlugin):
    '''Basic GAN discriminator.

    '''

    def build(self, input_shape, discriminator_args=None):
        '''Build discriminator model.

        Args:
            discriminator_args: discriminator arguments.

        '''
        Module = discriminator_args['Module']
        layers = discriminator_args['layers']

        layers = copy.deepcopy(layers)
        layers.append(dict(layer='linear', args=(1,)))
        self.add_nets(discriminator=Module(input_shape, layers=layers))

    def routine(self, X_P, X_Q, measure: str='JSD', gradient_penalty=1.0):
        '''

        Args:
            X_P: real input.
            X_Q: fake input.
            measure: Measure to compare real and fake distributions.
            gradient_penalty: Gradient penalty amount.

        '''

        E_pos, E_neg, P_samples, Q_samples = self.score(X_P, X_Q, measure)

        difference = E_pos - E_neg

        gp_loss_P = contrastive_gradient_penalty(self.nets.discriminator, X_P, gradient_penalty)
        gp_loss_Q = contrastive_gradient_penalty(self.nets.discriminator, X_Q, gradient_penalty)
        gp_loss = 0.5 * (gp_loss_P + gp_loss_Q)

        self.results.update(
            Scores={'E_P[D(x)]': P_samples.mean().item(),
                    'E_Q[D(x)]': Q_samples.mean().item()},
            Gradient_penalty=gp_loss.item()
        )
        if measure == 'JSD':
            measure_est = 0.5 * difference
        else:
            measure_est = 0.5 * difference

        self.add_results(**{measure + ' Est': measure_est.detach().item()})
        self.add_losses(discriminator=-difference + gp_loss)

    def score(self, X_P, X_Q, measure):
        '''Score real and fake.

        Args:
            X_P: Real tensor (unshuffled).
            X_Q: Fake tensor (shuffled).
            measure: Comparison measure.

        Returns:
            tuple of torch.Tensor: (real expectation, fake expectation, real samples, fake samples)

        '''
        discriminator = self.nets.discriminator
        P_samples = discriminator(X_P)
        Q_samples = discriminator(X_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)

        return E_pos, E_neg, P_samples, Q_samples

    def visualize(self, X_P, X_Q, measure=None):
        '''Visualize discriminator.
        '''
        E_pos, E_neg, P_samples, Q_samples = self.score(X_P, X_Q, measure)

        self.add_histogram(dict(fake=Q_samples.view(-1).data,
                                real=P_samples.view(-1).data),
                           name='Discriminator output')