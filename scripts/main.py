'''Deep Implicit Infomax

'''

import argparse
from argparse import RawTextHelpFormatter
import logging
import sys

from cortex.main import run

from cortex_DIM.evaluation_models.classification_eval import ClassificationEval
from cortex_DIM.evaluation_models.ndm_eval import NDMEval
from cortex_DIM.evaluation_models.msssim_eval import MSSSIMEval
from cortex_DIM.evaluation_models.mine_eval import MINEEval
from cortex_DIM.models.controller import Controller
from cortex_DIM.models.coordinates import CoordinatePredictor
from cortex_DIM.models.dim import GlobalDIM, LocalDIM
from cortex_DIM.models.prior_matching import PriorMatching


logger = logging.getLogger('DIM')


if __name__ == '__main__':

    mode_dict = dict(
        local=LocalDIM,
        glob=GlobalDIM,
        prior=PriorMatching,
        coordinates=CoordinatePredictor,
        classifier=ClassificationEval,
        ndm=NDMEval,
        mine=MINEEval,
        msssim=MSSSIMEval
    )

    names = tuple(mode_dict.keys())

    infos = []
    for k in mode_dict.keys():
        mode = mode_dict[k]
        info = mode.__doc__.split('\n', 1)[0]  # Keep only first line of doctstring.
        infos.append('{}: {}'.format(k, info))
    infos = '\n\t'.join(infos)

    models = []
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('models', nargs='+', choices=names,
                        help='Models used in Deep InfoMax. Choices are: \n\t{}'.format(infos))
    i = 1
    while True:
        arg = sys.argv[i]
        if arg in ('--help', '-h') and i == 1:
            i += 1
            break

        if arg.startswith('-'):
            break  # argument have begun
        i += 1
    args = parser.parse_args(sys.argv[1:i])
    models = args.models
    models = list(set(models))
    models = dict((k, mode_dict[k]) for k in models)

    sys.argv = [sys.argv[0]] + sys.argv[i:]
    controller = Controller(inputs=dict(inputs='data.images'), **models)
    run(controller)
