'''Deep Implicit Infomax

'''

import logging

from cortex.main import run
from cortex.plugins import ModelPlugin
from cortex.built_ins.models.classifier import SimpleClassifier

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from cortex_DIM.functions.dim_losses import fenchel_dual_loss, multi_donsker_varadhan_loss, nce_loss, \
    multi_nce_loss, donsker_varadhan_loss, multi_fenchel_dual_loss
from cortex_DIM.configs.convnets import configs as convnet_configs
from cortex_DIM.configs.resnets import configs as resnet_configs


logger = logging.getLogger('cortex_DIM')


class DIM(ModelPlugin):
    '''Deep InfoMax

    '''

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images'), skip_last_batch=True),
        train=dict(save_on_lowest='losses.encoder', epochs=1000),
        model=dict(
            classifier_c_args=dict(dropout=0.1, dim_h=[200], batch_norm=True),
            classifier_m_args=dict(dropout=0.1, dim_h=[200], batch_norm=True),
            classifier_f_args=dict(dropout=0.1, dim_h=[200], batch_norm=True),
            classifier_g_args=dict(dropout=0.1, dim_h=[200], batch_norm=True)),
        optimizer=dict(learning_rate=1e-4)
    )

    def __init__(self, Classifier=SimpleClassifier):
        super().__init__()

        self.classifier_c = Classifier(
            nets=dict(classifier='classifier_c'),
            kwargs=dict(classifier_args='classifier_c_args'))
        self.classifier_m = Classifier(
            nets=dict(classifier='classifier_m'),
            kwargs=dict(classifier_args='classifier_m_args'))
        self.classifier_f = Classifier(
            nets=dict(classifier='classifier_f'),
            kwargs=dict(classifier_args='classifier_f_args'))
        self.classifier_g = Classifier(
            nets=dict(classifier='classifier_g'),
            kwargs=dict(classifier_args='classifier_g_args'))

    def build(self, global_units=64, mi_units=1024, encoder_config='basic32x32',
              encoder_args={}):
        '''

        Args:
            global_units: Number of global units.
            mi_units: Number of units for MI estimation.
            encoder_config: Config of encoder. See `cortex_DIM/configs` for details.
            encoder_args: Additional dictionary to update encoder.

        '''

        # Draw data to help with shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()

        dim_c, dim_x, dim_y = self.get_dims('images')
        input_shape = (dim_x, dim_y, dim_c)

        # Create encoder.
        configs = dict()
        configs.update(**convnet_configs)
        configs.update(**resnet_configs)

        encoder_args_ = configs.get(encoder_config, None)
        if encoder_args_ is None:
            raise logger.warning('encoder_type `{}` not supported'.format(encoder_config))
            encoder_args_ = {}
        encoder_args_.update(**encoder_args)
        encoder_args = encoder_args_

        if global_units > 0:
            if 'fc_args' in list(encoder_args.keys()):
                encoder_args['fc_args'].append((global_units, False, False, None))
            else:
                encoder_args['fc_args'] = [(global_units, False, False, None)]
        else:
            if 'fc_args' in list(encoder_args.keys()):
                encoder_args.pop('fc_args')

        Encoder = encoder_args.pop('Encoder')
        self.nets.encoder = Encoder(input_shape, **encoder_args)

        # Create MI nn_modules and classifiers for monitoring.
        S = self.inputs('images').cpu()
        L, C, M, F, G = self.nets.encoder(S)

        local_units, locals_x, locals_y = L.size()[1:]
        self.nets.local_net = MI1x1ConvNet(local_units, mi_units)

        conv_units, conv_x, conv_y = C.size()[1:]
        self.classifier_c.build(dim_in=conv_units * conv_x * conv_y)

        if M is not None:
            multi_units, multis_x, multis_y = M.size()[1:]
            self.nets.multi_net = MI1x1ConvNet(multi_units, mi_units)
            self.classifier_m.build(dim_in=multi_units * multis_x * multis_y)

        if F is not None:
            fc_units = F.size(1)
            self.classifier_f.build(dim_in=fc_units)

        if G is not None:
            self.nets.global_net = MIFCNet(global_units, mi_units)
            self.classifier_g.build(dim_in=global_units)

    def routine(self, measure='JSD', mode='fd'):
        '''

        Args:
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

        '''
        X, Y = self.inputs('images', 'targets')
        L, C, M, F, G = self.nets.encoder(X)

        if G is not None:
            # Add a global-local loss.
            local_global_loss = self.local_global_loss(L, G, measure, mode)
            self.losses.global_net = local_global_loss
        else:
            local_global_loss = 0.

        if M is not None:
            # Add a multi-global local loss.
            local_multi_loss = self.local_multi_loss(L, M, measure, mode)
            self.losses.multi_net = local_multi_loss
        else:
            local_multi_loss = 0.

        self.losses.encoder = local_global_loss + local_multi_loss
        self.losses.local_net = local_global_loss + local_multi_loss

        # Classifiers
        units, dim_x, dim_y = C.size()[1:]
        C = C.view(-1, units * dim_x * dim_y)
        self.classifier_c.routine(C.detach(), Y)

        if M is not None:
            units, dim_x, dim_y = M.size()[1:]
            M = M.view(-1, units * dim_x * dim_y)
            self.classifier_m.routine(M.detach(), Y)

        if F is not None:
            self.classifier_f.routine(F.detach(), Y)

        if G is not None:
            self.classifier_g.routine(G.detach(), Y)

    def local_global_loss(self, l, g, measure, mode):
        '''

        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

        Returns:
            torch.Tensor: Loss.

        '''
        l_enc = self.nets.local_net(l)
        g_enc = self.nets.global_net(g)
        N, local_units, dim_x, dim_y = l_enc.size()
        l_enc = l_enc.view(N, local_units, -1)

        if mode == 'fd':
            loss = fenchel_dual_loss(l_enc, g_enc, measure=measure)
        elif mode == 'nce':
            loss = nce_loss(l_enc, g_enc)
        elif mode == 'dv':
            loss = donsker_varadhan_loss(l_enc, g_enc)
        else:
            raise NotImplementedError(mode)

        return loss

    def local_multi_loss(self, l, m, measure, mode):
        '''

        Args:
            l: Local feature map.
            m: Multiple globals feature map.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

        Returns:
            torch.Tensor: Loss.

        '''
        l_enc = self.nets.local_net(l)
        m_enc = self.nets.multi_net(m)
        N, local_units, dim_x, dim_y = l_enc.size()
        l_enc = l_enc.view(N, local_units, -1)
        m_enc = m_enc.view(N, local_units, -1)

        if mode == 'fd':
            loss = multi_fenchel_dual_loss(l_enc, m_enc, measure=measure)
        elif mode == 'nce':
            loss = multi_nce_loss(l_enc, m_enc)
        elif mode == 'dv':
            loss = multi_donsker_varadhan_loss(l_enc, m_enc)
        else:
            raise NotImplementedError(mode)

        return loss


if __name__ == '__main__':
    run(DIM())
