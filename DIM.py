'''Deep Implicit Infomax

'''

from cortex.main import run
from cortex.plugins import ModelPlugin
from cortex.built_ins.models.gan import SimpleDiscriminator, GradientPenalty, \
    get_positive_expectation, get_negative_expectation, generator_loss
from cortex.built_ins.networks.convnets import SimpleConvEncoder
import torch

from image_encoder import ImageEncoder


def random_permute(X):
    X = X.transpose(1, 2)
    b = torch.rand((X.size(0), X.size(1))).cuda()
    idx = b.sort(0)[1]
    adx = torch.range(0, X.size(1) - 1).long()
    X = X[idx, adx[None, :]].transpose(1, 2)

    return X


class DIM(ModelPlugin):
    '''Deep INFOMAX

    As featured in https://arxiv.org/abs/1808.06670

    '''
    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images'), skip_last_batch=True),
        train=dict(save_on_lowest='losses.topnet', epochs=1000),
        model=dict(discriminator_args=dict(batch_norm=False, dim_h=[1000, 200]),
                   encoder_args=dict(batch_norm=True,
                                     fully_connected_layers=1024,
                                     output_nonlinearity='sigmoid',
                                     min_dim=5),
                   prior_penalty_amount=0.5,
                   mine_measure='JSD',
                   classifier_args=dict(dropout=0.1, dim_h=[200], batch_norm=True),
                   classifier_h_args=dict(dropout=0.1, dim_h=[200], batch_norm=True),
                   classifier_c_args=dict(dropout=0.1, dim_h=[200], batch_norm=True))
    )

    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder(
            contract=dict(kwargs=dict(dim_out='dim_z')))
        self.discriminator = SimpleDiscriminator(
            contract=dict(kwargs=dict(dim_in='dim_z', measure='prior_measure')))
        self.penalty = GradientPenalty(
            contract=dict(nets=dict(network='discriminator'),
                          kwargs=dict(penalty_amount='prior_penalty_amount',
                                      penalty_type='prior_penalty_type')))

    def build(self, dim_z=64, noise_type='uniform', conv_idx=1):
        '''
        Args:
            dim_z: Size of the encoder output.
            noise_type: Noise type of the prior that matches to the output.
            conv_idx: Level to perform MI maximization on the convnet.
                0 is the top, -1 is the bottom.

        '''

        self.add_noise('Z', dist=noise_type, size=dim_z, low=0)
        self.encoder.build(dim_out=dim_z)
        self.discriminator.build(dim_in=dim_z)

        self.c_idx = self.encoder.conv_indices[conv_idx]
        X = self.inputs('inputs').cpu()
        self.nets.encoder(X, nonlinearity=False)
        C = self.nets.encoder.states[self.c_idx]
        dim_h, dim_x, dim_y = C.size()[1:]

        self.nets.topnet = SimpleConvEncoder(
            shape=(dim_x, dim_y, dim_h + dim_z),
            f_size=1, stride=1, dim_h=[512, 512, 1],
            pad=0, batch_norm=False, last_conv_nonlinearity=False)
        # Last key avoids last nonlinearity in convnet

    def routine(self, Z, prior_measure='GAN',
                prior_loss_type='non-saturating',
                beta=1.0, mine_measure='JSD'):
        '''

        Args:
            prior_loss_type: Adversarial loss type for the encoder.
                Used for the prior term only.
            beta: Amount of prior term for encoder
            prior_measure: Measure used for the prior matching.
            mine_measure: Measure used for the MI estimation.

        '''

        prior_scores = self.nets.discriminator(Z)
        prior_term = generator_loss(prior_scores, prior_measure,
                                    loss_type=prior_loss_type)

        Z = self.nets.encoder.states[-1] # without the output nonlinearity

        E_pos, E_neg, P_samples, Q_samples = self.score(Z, mine_measure)
        difference = E_pos - E_neg

        self.losses.encoder = -difference + beta * prior_term
        self.losses.topnet = -difference
        self.results.update(Scores=dict(Ep=P_samples.mean().item(),
                                        Eq=Q_samples.mean().item()))
        self.results[
            '{} distance'.format(mine_measure)] = difference.item()

    def score(self, Z, measure):
        P_scores = self.get_scores(Z)
        Q_scores = self.get_scores(Z, shuffle=True)

        E_pos = get_positive_expectation(P_scores, measure)
        E_neg = get_negative_expectation(Q_scores, measure)

        return E_pos, E_neg, P_scores, Q_scores

    def get_scores(self, z, shuffle=False):
        c = self.nets.encoder.states[self.c_idx + 2] # After the ReLU
        dim_x, dim_y = c.size(2), c.size(3)

        if shuffle:
            c = c.view(-1, c.size(1), dim_x * dim_y)
            c = random_permute(c)
            c = c.view(c.size(0), -1, dim_x, dim_y)

        z = z[:, :, None, None].expand(-1, -1, dim_x, dim_y)
        u = torch.cat([z, c], dim=1)
        y = self.nets.topnet(u)
        y = y.view(y.size(0), -1)

        return y

    def train_step(self):
        self.data.next()
        inputs_P, Z = self.inputs('inputs', 'Z')

        Z_Q = self.encoder.encode(inputs_P)
        self.discriminator.routine(Z, Z_Q)
        self.optimizer_step()

        self.penalty.routine(Z)
        self.optimizer_step()

        Z_Q = self.encoder.encode(inputs_P)
        self.routine(Z_Q)
        self.optimizer_step()

        self.encoder.routine(auto_input=True)
        self.optimizer_step()

    def eval_step(self):
        self.data.next()
        inputs_P, Z = self.inputs('inputs', 'Z')

        Z_Q = self.encoder.encode(inputs_P)

        self.discriminator.routine(Z, Z_Q)
        self.routine(Z_Q)
        self.encoder.routine(auto_input=True)

    def visualize(self, inputs, Z):
        Z_Q = self.encoder.encode(inputs)

        self.add_histogram(
            dict(real=Z.view(-1).data, fake=Z_Q.view(-1).data),
            name='encoder_output')


if __name__ == '__main__':
    run(DIM())
