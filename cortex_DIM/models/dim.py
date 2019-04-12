'''Deep infomax models.

'''

import copy

from cortex.plugins import ModelPlugin
import torch

from cortex_DIM.functions.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss
from cortex_DIM.nn_modules.convnet import Convnet
from cortex_DIM.nn_modules.resnet import ResNet
from cortex_DIM.nn_modules.mi_networks import MI1x1ConvNet, NopNet


def sample_locations(enc, n_samples):
    '''Randomly samples locations from localized features.

    Used for saving memory.

    Args:
        enc: Features.
        n_samples: Number of samples to draw.

    Returns:
        torch.Tensor

    '''
    n_locs = enc.size(2)
    batch_size = enc.size(0)
    weights = torch.tensor([1. / n_locs] * n_locs, dtype=torch.float)
    idx = torch.multinomial(weights, n_samples * batch_size, replacement=True) \
        .view(batch_size, n_samples)
    enc = enc.transpose(1, 2)
    adx = torch.arange(0, batch_size).long()
    enc = enc[adx[:, None], idx].transpose(1, 2)
    return enc


def compute_dim_loss(l_enc, m_enc, measure, mode):
    '''Computes DIM loss.

    Args:
        l_enc: Local feature map encoding.
        m_enc: Multiple globals feature map encoding.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

    Returns:
        torch.Tensor: Loss.

    '''

    if mode == 'fd':
        loss = fenchel_dual_loss(l_enc, m_enc, measure=measure)
    elif mode == 'nce':
        loss = infonce_loss(l_enc, m_enc)
    elif mode == 'dv':
        loss = donsker_varadhan_loss(l_enc, m_enc)
    else:
        raise NotImplementedError(mode)

    return loss


class GlobalDIM(ModelPlugin):
    '''Global version of Deep InfoMax

    '''
    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='data.images'), skip_last_batch=True),
        train=dict(save_on_lowest='losses.encoder', epochs=1000),
        optimizer=dict(learning_rate=1e-4)
    )

    def build(self, encoder, config, task_idx=None, mi_units=2048):
        '''

        Args:
            task_idx (tuple): Indices where to do local objective.
            mi_units (int): Number of units for MI estimation.

        '''

        self.nets.encoder = encoder

        if task_idx is not None:
            self.task_idx = task_idx
        elif 'local_task_idx' not in config.keys():
            raise ValueError('')
        else:
            self.task_idx = config['local_task_idx']

        local_idx, global_idx = self.task_idx

        # Create MI nn_modules.
        X = self.inputs('data.images')
        outs = self.nets.encoder(X, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]
        local_size = L.size()[1:]
        global_size = G.size()[1:]

        # For global DIM, we'll just copy the layer hyperparameters for the encoder.
        layers = copy.deepcopy(config['layers'])
        layers[-1] = dict(layer='linear', args=(mi_units,))

        if isinstance(encoder.module.encoder, ResNet):
            Encoder = ResNet
        elif isinstance(encoder.module.encoder, Convnet):
            Encoder = Convnet
        else:
            raise NotImplementedError('Can\'t do {} yet (feature request)'.format(type(encoder.encoder)))

        if len(local_size) == 1 or len(local_size) == 3:
            local_MINet = Encoder(local_size[::-1], layers=layers[local_idx:])
        else:
            raise NotImplementedError()

        if len(global_size) == 1:
            if global_size[0] == mi_units:
                global_MINet = NopNet()
            else:
                global_MINet = Encoder(global_size, layers=layers[global_idx:])
        elif len(global_size) == 3:
            if (global_size[1] == global_size[1] == 1) and global_size[0] == mi_units:
                global_MINet = NopNet()
            else:
                global_MINet = Encoder(global_size, layers=layers[global_idx:])
        else:
            raise NotImplementedError()

        local_MINet = local_MINet.to(X.device)
        global_MINet = global_MINet.to(X.device)

        def extract(outs, local_net=None, global_net=None):
            '''Wrapper function to be put in encoder forward for speed.

            Args:
                outs (list): List of activations
                local_net (nn.Module): Network to encode local activations.
                global_net (nn.Module): Network to encode global activations.

            Returns:
                tuple: local, global outputs

            '''
            l_idx, g_idx = self.task_idx
            L = outs[l_idx]
            G = outs[g_idx]

            L = local_net(L)
            G = global_net(G)

            N, local_units = L.size()[:2]
            L = L.view(N, local_units, -1)
            G = G.view(N, local_units, -1)

            return L, G

        self.nets.encoder.module.add_network(self.name, extract,
                                             networks=dict(local_net=local_MINet, global_net=global_MINet))

    def routine(self, outs=None, measure='JSD', mode='fd', scale=1.0, act_penalty=0.):
        '''

        Args:
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
            scale: Hyperparameter for global DIM. Called `alpha` in the paper.
            act_penalty: L2 penalty on the global activations.

        '''
        L, G = outs[self.name]

        act_loss = (G ** 2).sum(1).mean()

        loss = compute_dim_loss(L, G, measure, mode)

        if scale > 0:
            self.add_losses(encoder=scale * loss + act_penalty * act_loss)


class LocalDIM(ModelPlugin):
    '''Local version of Deep InfoMax

    '''

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='data.images'), skip_last_batch=True),
        train=dict(save_on_lowest='losses.encoder', epochs=1000),
        optimizer=dict(learning_rate=1e-4)
    )

    def build(self, encoder, config, task_idx=None, mi_units=2048, global_samples=None, local_samples=None):
        '''

        Args:
            global_units: Number of global units.
            task_idx (tuple): Indices where to do local objective.
            mi_units: Number of units for MI estimation.
            global_samples: Number of samples over the global locations for each example.
            local_samples: Number of samples over the local locations for each example.

        '''
        self.nets.encoder = encoder

        if task_idx is not None:
            self.task_idx = task_idx
        elif 'local_task_idx' not in config.keys():
            raise ValueError('No task_idx provided for local task.')
        else:
            self.task_idx = config['local_task_idx']

        # Create MI nn_modules.
        X = self.inputs('data.images')
        outs = self.nets.encoder(X, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]
        local_size = L.size()[1:]
        global_size = G.size()[1:]

        if len(local_size) == 1 or len(local_size) == 3:
            local_MINet = MI1x1ConvNet(local_size[0], mi_units)
        else:
            raise NotImplementedError()

        if len(global_size) == 1:
            if global_size[0] == mi_units:
                global_MINet = NopNet()
            else:
                global_MINet = MI1x1ConvNet(global_size[0], mi_units)
        elif len(global_size) == 3:
            if (global_size[1] == global_size[1] == 1) and global_size[0] == mi_units:
                global_MINet = NopNet()
            else:
                global_MINet = MI1x1ConvNet(global_size[0], mi_units)
        else:
            raise NotImplementedError()

        local_MINet = local_MINet.to(X.device)
        global_MINet = global_MINet.to(X.device)

        def extract(outs, local_net=None, global_net=None):
            '''Wrapper function to be put in encoder forward for speed.

            Args:
                outs (list): List of activations
                local_net (nn.Module): Network to encode local activations.
                global_net (nn.Module): Network to encode global activations.

            Returns:
                tuple: local, global outputs

            '''
            l_idx, g_idx = self.task_idx
            L = outs[l_idx]
            G = outs[g_idx]

            # All globals are reshaped as 1x1 feature maps.
            global_size = G.size()[1:]
            if len(global_size) == 1:
                G = G[:, :, None, None]

            L = local_net(L)
            G = global_net(G)

            N, local_units = L.size()[:2]
            L = L.view(N, local_units, -1)
            G = G.view(N, local_units, -1)

            # Sample locations for saving memory.
            if global_samples is not None:
                G = sample_locations(G, global_samples)

            if local_samples is not None:
                L = sample_locations(L, local_samples)

            return L, G

        self.nets.encoder.module.add_network(self.name, extract,
                                             networks=dict(local_net=local_MINet, global_net=global_MINet))

    def routine(self, outs=None, measure='JSD', mode='fd', scale=1.0, act_penalty=0.):
        '''

        Args:
            measure: Type of f-divergence. For use with mode `fd`.
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
            scale: Hyperparameter for local DIM. Called `beta` in the paper.
            act_penalty: L2 penalty on the global activations. Can improve stability.

        '''
        L, G = outs[self.name]

        if act_penalty > 0.:
            act_loss = act_penalty * (G ** 2).sum(1).mean()
        else:
            act_loss = 0.

        loss = compute_dim_loss(L, G, measure, mode)

        if scale > 0:
            self.add_losses(encoder=scale * loss + act_loss)
