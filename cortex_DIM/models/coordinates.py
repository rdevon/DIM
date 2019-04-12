'''Coordinate task

'''

from cortex.plugins import ModelPlugin
import torch
import torch.nn.functional as F

from cortex_DIM.nn_modules.mi_networks import MI1x1ConvNet


class CoordinatePredictor(ModelPlugin):
    '''Coordinate prediction

    '''
    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='data.images'), skip_last_batch=True),
        train=dict(save_on_lowest='losses.encoder', epochs=1000),
        optimizer=dict(learning_rate=1e-4)
    )

    def build(self, encoder, config, task_idx=None):
        '''

        Args:
            task_idx: Indices for coordinate task.

        '''

        self.nets.encoder = encoder

        if task_idx is not None:
            self.task_idx = task_idx
        elif 'local_task_idx' not in config.keys():
            raise ValueError('')
        else:
            self.task_idx = config['local_task_idx']

        # Create MI nn_modules.
        X = self.inputs('data.images')
        outs = self.nets.encoder(X, return_all_activations=True)
        L, G = [outs[i] for i in self.task_idx]
        local_size = L.size()[1:]
        dim_x = local_size[1]
        dim_y = local_size[2]
        n_coords = dim_x + dim_y
        global_size = G.size()[1:]
        n_inputs = global_size[0] + local_size[0]
        if len(global_size) != 1:
            raise NotImplementedError('Global vector must be 1d')

        # Set up ground truth labels
        self.labels = torch.zeros((n_coords, dim_x, dim_y)).float().to(L.device)
        for i in range(dim_x):
            for j in range(dim_y):
                self.labels[i, i, j] = 1.
                self.labels[dim_x + j, i, j] = 1.

        coord_net = MI1x1ConvNet(n_inputs, n_coords).to(X.device)

        def extract(outs, coord_net=None):
            '''Wrapper function to be put in encoder forward for speed.

            Args:
                outs (list): List of activations
                coord_net (nn.Module): Network to predict coordinates of every location.

            Returns:
                tuple: local, global outputs

            '''
            L, G = [outs[i] for i in self.task_idx]

            input = torch.cat([L, G[:, :, None, None].expand(-1, -1, L.size(2), L.size(3))], dim=1)
            logits = coord_net(input)

            return logits

        self.nets.encoder.module.add_network(self.name, extract,
                                             networks=dict(coord_net=coord_net))

    def routine(self, outs=None, scale=1.0):
        '''

        Args:
            scale: Scaling term for loss on the encoder.

        '''
        logits = outs[self.name]

        labels_ex = self.labels[None, :, :, :].expand(logits.size(0), -1, -1, -1)

        x_logits, y_logits = torch.chunk(logits, 2, dim=1)
        x_labels, y_labels = torch.chunk(labels_ex, 2, dim=1)

        x_sm_out = F.log_softmax(x_logits, dim=1)
        y_sm_out = F.log_softmax(y_logits, dim=1)

        x_loss = -(x_labels * x_sm_out).sum(1).mean()
        y_loss = -(y_labels * y_sm_out).sum(1).mean()
        loss = x_loss + y_loss

        # Computing accuracies.
        x_labels = torch.max(x_labels.data, 1)[1]
        y_labels = torch.max(y_labels.data, 1)[1]

        x_pred = torch.max(x_logits.data, 1)[1]
        y_pred = torch.max(y_logits.data, 1)[1]

        x_correct = 100. * x_pred.eq(x_labels.data).float().cpu().mean()
        y_correct = 100. * y_pred.eq(y_labels.data).float().cpu().mean()
        self.add_losses(encoder=scale * loss)
        self.add_results(x_accuracy=x_correct, y_accuracy=y_correct, total_accuracy=0.5 * (x_correct + y_correct))
