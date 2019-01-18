'''Script classification.

'''

import logging

from cortex.main import run
from cortex.plugins import ModelPlugin

from cortex_DIM.models.classifier import Classifier


logger = logging.getLogger('DIM_evaluation')


class ClassificationEvaluator(ModelPlugin):
    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images'), skip_last_batch=True),
        optimizer=dict(learning_rate=1e-4,
                       scheduler='MultiStepLR',
                       scheduler_options=dict(milestones=[50,100],gamma=0.1))
    )
    def build(self, encoder_key='encoder', classification_idx=[1, 3, 4],
              classifier_kwargs=[dict(fc_args=[(200, True, 0.1, 'ReLU')]),
                                 dict(fc_args=[(200, True, 0.1, 'ReLU')]),
                                 dict(fc_args=[(200, True, 0.1, 'ReLU')])],
              classifier_names=['local', 'fully_connected', 'global']):
        '''Builds the classifiers.

        Args:
            encoder_key: Dictionary key for the encoder.
            classification_idx: Indices of encoder output to classify.
            classifier_kwargs: kwargs for different classifiers.
            names: Names for classifier.

        '''

        # Fetch the encoder and remove everything else.
        encoder = None
        for k in self.nets.keys():
            if k != encoder_key:
                model = self.nets.pop(k)
                del model
            else:
                encoder = self.nets.pop(k)

        if encoder is None:
            raise KeyError('{} not found in loaded model.'.format(encoder_key))

        self.nets.encoder = encoder

        # Draw data to help with shape inference.
        self.data.reset(mode='test', make_pbar=False)
        self.data.next()

        n_labels = self.get_dims('targets')
        X = self.inputs('images').cpu()
        outs = self.nets.encoder(X)

        self.classification_idx = classification_idx
        self.classifier_names = classifier_names

        for i, idx in enumerate(self.classification_idx):
            name = self.classifier_names[i]
            contract = dict(nets=dict(classifier='classifier_{}'.format(name)))
            setattr(self, 'classifier_{}'.format(name), Classifier(**contract))
            classifier = getattr(self, 'classifier_{}'.format(name))
            input_shape = outs[idx].size()[1:]
            kwargs = classifier_kwargs[i]
            classifier.build(input_shape, n_labels, **kwargs)

    def routine(self, inputs, targets):
        '''Classification routine.
        '''
        for i, idx in enumerate(self.classification_idx):
            name = self.classifier_names[i]
            classifier = getattr(self, 'classifier_{}'.format(name))
            output = self.nets.encoder(inputs)[idx]
            classifier.routine(output.detach(), targets)

    def visualize(self, inputs, targets):
        '''Visualization.
        '''
        for i, idx in enumerate(self.classification_idx):
            name = self.classifier_names[i]
            classifier = getattr(self, 'classifier_{}'.format(name))
            output = self.nets.encoder(inputs)[idx]
            classifier.visualize(inputs, output, targets)


if __name__ == '__main__':
    run(ClassificationEvaluator())
