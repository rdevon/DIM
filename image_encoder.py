'''Simple Image encoder mmdel.

'''

import cortex
from cortex.plugins import ModelPlugin
from cortex.built_ins.models.utils import update_encoder_args, update_decoder_args
from cortex.built_ins.models.classifier import SimpleClassifier, SimpleAttributeClassifier
from cortex.built_ins.networks.resnets import ResBlock
import torch


from alexnet_64 import AlexNetEncoder


class EncoderBase(ModelPlugin):

    def build(self, encoder_type: str='convnet', dim_out: int=None,
              encoder_args=dict(), semi_supervised=None):
        '''

        Args:
            encoder_type: Encoder model type.
            dim_out: Output size.
            encoder_args: Arguments for encoder build.

        '''
        x_shape = self.get_dims('x', 'y', 'c')
        if 'encoder' not in self.nets:
            if encoder_type == 'alexnet':
                Encoder = AlexNetEncoder
                encoder_args.pop('min_dim')
                encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)
            else:
                Encoder, _ = update_encoder_args(
                    x_shape, model_type=encoder_type)
                encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)

            self.nets.encoder = encoder

        self.data.reset(mode='test', make_pbar=False)
        self.data.next()
        X = self.inputs('inputs').cpu()
        Z = self.nets.encoder(X)
        dim_z = Z.size(1)

        self.linear_indices = []
        self.conv_indices = []
        self.c_idx = None

        index = -2
        try:
            while True:
                layer = self.nets.encoder.models[index]
                if isinstance(layer, torch.nn.modules.linear.Linear):
                    # Is a linear layer.
                    self.linear_indices.append(index)
                elif isinstance(layer, cortex.built_ins.networks.modules.View):
                    # Is a flattened version of a convolutional layer.
                    self.c_idx = self.c_idx or index
                elif isinstance(layer, torch.nn.modules.conv.Conv2d):
                    # Is a convolutional layer.
                    self.conv_indices.append(index)
                index -= 1
        except IndexError:
            pass

        if not semi_supervised:
            # Build the classifier on top of Y.
            self.classifier.build(dim_in=dim_z)

            # Build the classifier on top of the layer below Y.
            self.h_idx = self.linear_indices[0]
            dim_h = self.nets.encoder.states[self.h_idx].size(1)
            self.classifier_h.build(dim_in=dim_h)

        # Build the classifier on top of the last conv layer.
        dim_c = self.nets.encoder.states[self.c_idx].size(1)
        self.classifier_c.build(dim_in=dim_c)

    def encode(self, inputs, **kwargs):
        return self.nets.encoder(inputs, **kwargs)


class ImageEncoder(EncoderBase):
    '''Builds a simple image encoder.

    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = SimpleClassifier(
            contract=dict(kwargs=dict(dim_in='dim_z')))
        self.classifier_h = SimpleClassifier(
            contract=dict(nets=dict(classifier='classifier_h'),
                          kwargs=dict(classifier_args='classifier_h_args')))
        self.classifier_c = SimpleClassifier(
            contract=dict(nets=dict(classifier='classifier_c'),
                          kwargs=dict(classifier_args='classifier_c_args')))

    def routine(self, inputs, targets, semi_supervised=False):
        '''

        Args:
            semi_supervised: If set, pass classification gradients through
                encoder (off by default).

        '''
        Z_Q = self.encode(inputs, nonlinearity=False).detach()

        if semi_supervised:
            self.classifier_c.routine(
                self.nets.encoder.states[self.c_idx], targets)
            if 'classifier_c' in self.losses:
                # For STL, some batches will not have labels, so no loss.
                if 'encoder' in self.losses:
                    self.losses.encoder += self.losses.classifier_c
                else:
                    self.losses.encoder = self.losses.classifier_c

        else:
            self.classifier.routine(Z_Q.detach(), targets)
            self.classifier_h.routine(
                self.nets.encoder.states[self.h_idx].detach(), targets)
            self.classifier_c.routine(
                self.nets.encoder.states[self.c_idx].detach(), targets)
