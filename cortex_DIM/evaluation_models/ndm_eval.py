'''NDM evaluator model for encoder

'''

from cortex.main import run
import torch.nn as nn

from cortex_DIM.models.ndm import NDM


class NDMEval(NDM):
    '''Neural dependency measure evaluation module.

    Measures the KL divergence between representation and shuffled represention distributions.

    '''
    defaults = dict(
        data=dict(batch_size=dict(train=10000, test=10000),
                  inputs=dict(inputs='images'),
                  skip_last_batch=True),
        optimizer=dict(learning_rate=1e-4,
                       scheduler='MultiStepLR',
                       scheduler_options=dict(milestones=[500, 750], gamma=0.1))
    )

    def build(self, encoder, config, task_idx=-1,
              layers=[dict(layer='linear', args=(1000,), act='ReLU'),
                      dict(layer='linear', args=(200,), act='ReLU')]):
        '''Builds NDM evaluator.

        Args:
            encoder_key: Dictionary key for the encoder.
            task_idx: Index of tensor to measure NDM.
            layers: NDM layer arguments.

        '''
        self.nets.encoder = encoder

        X = self.inputs('data.images')
        self.task_idx = task_idx
        out = encoder(X, return_all_activations=True)[self.task_idx]
        super().build(out.size()[1:], layers=layers)

    def routine(self, outs=None, measure='KL', nonlinearity=''):
        '''

        Args:
            measure: Type of measure to use for NDM.
            nonlinearity: Nonlinearity to use on output of encoder.

        Returns:

        '''
        if outs is None:
            inputs = self.inputs('data.images')
            outs = self.nets.encoder(inputs, return_all_activations=True)

        out = outs[self.task_idx]
        if nonlinearity != '':
            out = getattr(nn, nonlinearity)()(out)
        super().routine(out.detach(), measure=measure)

    def visualize(self, inputs, nonlinearity=None, measure=None):
        out = self.nets.encoder(inputs, return_all_activations=True)[self.task_idx]
        if nonlinearity != '':
            out = getattr(nn, nonlinearity)()(out)
        super().visualize(out, measure=measure)


if __name__ == '__main__':
    run(NDMEval())