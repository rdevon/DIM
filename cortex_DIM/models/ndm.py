'''Neural dependency measure (NDM).

'''

import torch

from cortex.plugins import ModelPlugin

from cortex_DIM.nn_modules.convnet import Convnet
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation


def random_permute(X):
    '''Randomly permuts a tensor along the batch axis independently for each feature.

    Args:
        X: Input tensor.

    Returns:
        torch.Tensor: shuffled tensor.

    '''
    b = torch.rand(X.size()).cuda()
    idx = b.sort(0)[1]
    adx = torch.arange(0, X.size(1)).long()
    return X[idx, adx[None, :]]


class NDM(ModelPlugin):
    '''Neural dependency measure.

    Measures the total correlation.

    '''

    def build(self, input_shape, layers=None):
        '''Build NDM model.

        Args:
            layers: NDM layers.

        '''
        layers.append(dict(layer='linear', args=(1,)))
        self.add_nets(ndm=Convnet(input_shape, layers=layers))

    def routine(self, Z_P, measure: str='KL'):
        '''

        Args:
            Z_P: Input unshuffled tensor.
            ndm_measure: Measure to compare representations with shuffled versions.

        '''
        Z_Q = random_permute(Z_P)
        E_pos, E_neg, P_samples, Q_samples = self.score(Z_P.detach(), Z_Q.detach(), measure)
        difference = E_pos - E_neg
        if measure == 'DV':
            ndm = E_pos - E_neg
        else:
            ndm = get_positive_expectation(P_samples, 'DV') - get_negative_expectation(Q_samples, 'DV')
            self.add_results(**{measure: difference.detach().item()})

        self.add_results(
            Scores={'E_P[D(x)]': P_samples.mean().detach().item(),
                    'max(D(x))': P_samples.max().detach().item(),
                    'E_Q[D(x)]': Q_samples.mean().detach().item()},
            NDM=ndm.detach().item())

        self.add_losses(ndm=-difference)

    def score(self, X_P, X_Q, measure):
        '''Score real and fake.

        Args:
            X_P: Real tensor (unshuffled).
            X_Q: Fake tensor (shuffled).
            measure: Comparison measure.

        Returns:
            tuple of torch.Tensor: (real expectation, fake expectation, real samples, fake samples)

        '''
        ndm = self.nets.ndm
        P_samples = ndm(X_P)
        Q_samples = ndm(X_Q)

        E_pos = get_positive_expectation(P_samples, measure)
        E_neg = get_negative_expectation(Q_samples, measure)

        return E_pos, E_neg, P_samples, Q_samples

    def visualize(self, Z_P, measure=None):
        '''Visualize NDM.
        '''
        Z_Q = random_permute(Z_P)

        E_pos, E_neg, P_samples, Q_samples = self.score(Z_P, Z_Q, measure)

        self.add_histogram(dict(fake=Q_samples.view(-1).data,
                                real=P_samples.view(-1).data),
                           name='NDM output')
