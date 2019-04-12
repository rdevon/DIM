'''Encoder eval for MS-SSIM

'''

from cortex.main import run

from cortex_DIM.configs.deconvnets import configs as decoder_configs
from cortex_DIM.models.decoder import Decoder


class MSSSIMEval(Decoder):
    '''Measure MS-SSIM through a decoder trained with reconstruction.

    '''
    defaults = dict(
        data=dict(batch_size=dict(train=64, test=64),
                  inputs=dict(inputs='images'),
                  skip_last_batch=True),
        optimizer=dict(learning_rate=1e-4,
                       scheduler='MultiStepLR',
                       scheduler_options=dict(milestones=[50, 100], gamma=0.1))
    )

    def build(self, encoder, config_,
              task_idx=-1, config='basic32x32', args={}):
        '''Builds MINE evaluator.

        Args:
            encoder_key: Dictionary key for the encoder.
            task_idx: Index of output tensor to measure MS-SSIM.
            config: Config name for decoder. See `configs` for details.
            args: Arguments to update config with.

        '''

        self.nets.encoder = encoder

        X = self.inputs('data.images')
        self.task_idx = task_idx
        out = self.nets.encoder(X, return_all_activations=True)[self.task_idx]

        config = decoder_configs.get(config)
        config.update(**args)

        super().build(out.size()[1:], args=config)

    def routine(self, outs=None):
        X = self.inputs('data.images')
        if outs is None:
            outs = self.nets.encoder(X, return_all_activations=True)

        out = outs[self.task_idx]
        super().routine(X, out.detach())

    def visualize(self, inputs):
        out = self.nets.encoder(inputs, return_all_activations=True)[self.task_idx]
        super().visualize(out)


if __name__ == '__main__':
    run(MSSSIMEval())