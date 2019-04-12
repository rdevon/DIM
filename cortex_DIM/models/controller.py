'''Simple model for combining models into a single encoder.

This helps for memory / speed for multi-gpu settings.

'''

import copy
import logging

from cortex.plugins import ModelPlugin
import torch

from cortex_DIM.configs.convnets import configs as convnet_configs
from cortex_DIM.configs.resnets import configs as resnet_configs
from cortex_DIM.configs.mlp import configs as mlp_configs
from cortex_DIM.models.dim import LocalDIM, GlobalDIM
from cortex_DIM.models.coordinates import CoordinatePredictor

logger = logging.getLogger('cortex_DIM')

# We put this out here instead of inside the encoder because of pickle.
_extract_fns = dict()


class BigEncoder(torch.nn.Module):
    '''A wrapping model that incorporates other self-supervision tasks.

    This is for speed and efficiency, particularly for multiple GPUs.

    Attributes:
        encoder (torch.nn.Module): Encoder network.
        extract_fns (dict): Dictionary of functions used to extract features for self-supervision models.

    '''

    def __init__(self, encoder):
        '''

        Args:
            encoder (torch.nn.Module): Encoder network.
        '''

        super().__init__()
        self.encoder = encoder
        self.network_keys = dict()

    def add_network(self, key, extract_fn, networks=None):
        ''' Adds a network for self-supervision.

        Args:
            key (str): Name of the task.
            extract_fn (callable): Function that extracts features for computing loss.
            local (torch.nn.Module): Network for local features.
            glob (torch.nn.Module): Network for global features.

        '''
        networks = networks or {}
        _extract_fns[key] = extract_fn
        self.network_keys[key] = []
        # Adds networks as encoder attributes for optimization related reasons.
        for k, v in networks.items():
            name = '{}_{}'.format(key, k)
            setattr(self, name, v)
            self.network_keys[key].append(k)

    def clear(self):
        '''Clears everything but encoder.

        '''
        if not hasattr(self, 'network_keys'):
            return  # This is for compatibility with an earlier version of the BigEncoder
        for key, v in self.network_keys.items():
            for k in v:
                name = '{}_{}'.format(key, k)
                delattr(self, name)

    def forward(self, X, return_rkhs=False, return_all_activations=False):
        '''Forward pass.

        Args:
            X (torch.Tensor): Input tensor.
            return_rkhs (bool): Return both layers and self-supervision network outputs.
            return_all_activations: Returns all activations for encoder.

        Returns:
            dict or (dict, dict)

        '''
        outs = self.encoder(X, return_full_list=return_all_activations)
        layer_outs = {}

        if return_all_activations:
            if return_rkhs:
                for key, extract_fn in _extract_fns.items():
                    # Grab the networks and pass these back through extract fn (necessary for DataParallel)
                    network_keys = self.network_keys[key]
                    networks = dict((k, getattr(self, '{}_{}'.format(key, k))) for k in network_keys)
                    layer_outs[key] = extract_fn(outs, **networks)

                return outs, layer_outs
            else:
                return outs
        else:
            return outs


class Controller(ModelPlugin):
    '''Model for managing semi-supervised losses and probes into encoder.

    '''
    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images')),
        train=dict(save_on_lowest='losses.encoder', epochs=1000),
        optimizer=dict(learning_rate=1e-4)
    )

    def __init__(self, inputs=None, **model_classes):
        '''Initializes the controller.

        Args:
            inputs (dict): Mapping for data names.
            **model_classes: name, class keyword pairs of models.
        '''

        super().__init__(inputs=inputs)
        for k, model_cls in model_classes.items():
            setattr(self, k, model_cls(inputs=inputs))

        self.model_names = list(model_classes.keys())

    def build(self, output_units=64, encoder_config='basic32x32', encoder_args={},
              eval=False):
        '''Builds the controller for multi-task DIM.

        Args:
            output_units: Number of global output units of encoder.
            encoder_config (str): Config name for encoder.
            encoder_args: Extra arguments for encoder.
            eval: Eval mode. Erases all nets but encoder.

        '''

        # Reset the data iterator and draw batch to perform shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()

        dim_c, dim_x, dim_y = self.get_dims('data.images')
        input_shape = (dim_x, dim_y, dim_c)

        # Draw the network hyperparameters from the config file.
        config_dict = dict()
        config_dict.update(**convnet_configs)
        config_dict.update(**resnet_configs)
        config_dict.update(**mlp_configs)

        config = config_dict.get(encoder_config, None)
        if config is None:
            raise logger.warning('encoder_type `{}` not supported'.format(encoder_config))
            config = {}
            config.update(**encoder_args)

        # Build the encoder.
        Encoder = config.pop('Module')
        task_keys = ['local_task_idx', 'global_task_idx', 'classifier_idx']

        if '{}.encoder'.format(self.name) in self._all_nets._loaded.keys():
            # Reloading the encoder.
            encoder = self._all_nets._loaded.pop('{}.encoder'.format(self.name))
            encoder.clear()
            if eval:
                net_keys = list(self._all_nets._loaded.keys())
                for k in net_keys:
                    self._all_nets._loaded.pop(k)
            self.nets.encoder = encoder
        else:
            # Build the encoder from scratch.
            encoder_args_ = dict((k, v) for k, v in config.items() if k not in task_keys)
            encoder_args_.update(**encoder_args)
            encoder_args = encoder_args_

            # Optionally add an output layer to the encoder layers.
            if output_units > 0:
                top_layer = dict(layer='linear', args=(output_units,))
                encoder_args['layers'].append(top_layer)
            encoder = BigEncoder(Encoder(input_shape, **encoder_args))
            self.add_nets(encoder=encoder)  # Adds the encoder to the list of networks.

        # Build the self-supervision and monitoring models.
        for name in self.model_names:
            model = getattr(self, name)
            model.build(self.nets.encoder, copy.deepcopy(config))

    def train_step(self, eval=None):
        '''Training step.

        Args:
            eval (bool): Eval mode.

        '''

        # Draw data
        self.data.next()
        X = self.inputs('data.images')

        # Forward pass through encoder.
        if eval: # We aren't going to use grads for encoder for eval, this helps save memory.
            with torch.no_grad():
                outs, layer_outs = self.nets.encoder(X, return_all_activations=True, return_rkhs=True)
        else:
            outs, layer_outs = self.nets.encoder(X, return_all_activations=True, return_rkhs=True)

        # Pass data to model routines. This collects all the losses and results from modules.
        for name in self.model_names:
            model = getattr(self, name)
            # Some models have their forward passes embedded into the encoder for speed.
            if isinstance(model, (LocalDIM, GlobalDIM, CoordinatePredictor)):
                model.routine(outs=layer_outs)
            else:
                model.routine(outs=outs)

        # Optimization step.
        self.optimizer_step()

    def eval_step(self):
        '''Evaluation step.

        '''

        # Draw data
        self.data.next()
        X = self.inputs('data.images')

        # Forward pass through encoder.
        outs, layer_outs = self.nets.encoder(X, return_all_activations=True, return_rkhs=True)

        # Pass data to model routines.
        for name in self.model_names:
            model = getattr(self, name)
            # Some models have their forward passes embedded into the encoder for speed.
            if isinstance(model, (LocalDIM, GlobalDIM, CoordinatePredictor)):
                model.routine(outs=layer_outs)
            else:
                model.routine(outs=outs)

    def visualize(self):
        X = self.inputs('data.images')
        self.add_image(X, name='Ground Truth')  # Visualizes the input image.
        for name in self.model_names:
            model = getattr(self, name)
            model.visualize(auto_input=True)
