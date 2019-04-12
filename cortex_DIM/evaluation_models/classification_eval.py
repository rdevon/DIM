'''Module for evaluating an encoder using classification.

'''

import copy

from cortex.plugins import ModelPlugin
import torch

from cortex_DIM.models.classifier import Classifier


class ClassificationEval(ModelPlugin):
    '''Basic classification module.

    This module forms several classifiers which plug into the encoder.

    '''

    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images')),
        optimizer=dict(learning_rate=1e-4,
                       scheduler='MultiStepLR',
                       scheduler_options=dict(milestones=[50, 100], gamma=0.1))
    )

    def build(self, encoder, config, task_idx={},
              layers=[dict(layer='linear', args=(200,), bn=True, do=0.1, act='ReLU', bias=True)]):
        '''Builds the classifiers.

        Args:
            task_idx: Dictionary of (name, index) pairs of encoder output to classify.
            layers: Layers for classifiers.

        '''
        self.nets.encoder = encoder

        if task_idx != {}:
            self.task_idx = task_idx
        elif config is None:
            raise ValueError('config and classifier_task_idx not provided.')
        elif 'classifier_idx' not in config.keys():
            raise ValueError('classifier_idx must be provided in config')
        else:
            self.task_idx = config['classifier_idx']

        n_labels = self.get_dims('data.targets')
        X = self.inputs('data.images')
        with torch.no_grad():
            outs = self.nets.encoder(X, return_all_activations=True)

        # Build all the classifiers
        classifier_task_idx_ = {}
        for name, task_idx in self.task_idx.items():
            out = outs[task_idx]
            input_shape = out.size()[1:]
            cls_name = '{}[{}]{}'.format(name, task_idx, tuple(input_shape))
            cls_name = cls_name.replace(' ', '')
            classifier_task_idx_[cls_name] = task_idx
            contract = dict(nets=dict(classifier=cls_name))
            self.add_model(cls_name, Classifier(**contract))
            classifier = getattr(self, cls_name)
            layers = copy.deepcopy(layers)
            classifier.build(input_shape, n_labels, layers=layers)
        self.task_idx = classifier_task_idx_

    def routine(self, outs=None, semi_supervised_task_idx=[]):
        '''Classification routine.

        Args:
            semi_supervised_task_idx: indices for backpropagating classification loss through encoder.

        '''
        targets = self.inputs('data.targets')
        if outs is None:
            inputs = self.inputs('data.images')
            outs = self.nets.encoder(inputs, return_all_activations=True)

        classification_losses = []
        for name, task_idx in self.task_idx.items():
            classifier = getattr(self, name)
            output = outs[task_idx]
            if semi_supervised_task_idx and (task_idx in semi_supervised_task_idx):
                input = output
            else:
                input = output.detach()

            classifier.routine(input, targets)

            if semi_supervised_task_idx and (task_idx in semi_supervised_task_idx):
                classification_losses.append(classifier.losses['classifier'])

        if semi_supervised_task_idx:
            self.losses.encoder = sum(classification_losses)

def visualize(self):
        '''Visualization.

        '''
        inputs, targets = self.inputs('data.images', 'data.targets')
        out = self.nets.encoder(inputs, return_all_activations=True)
        for name, task_idx in self.task_idx.items():
            classifier = getattr(self, name)
            output = out[task_idx]
            classifier.visualize(inputs, output, targets)
