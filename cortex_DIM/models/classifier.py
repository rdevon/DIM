'''Classification models.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.plugins import ModelPlugin

from cortex_DIM.nn_modules.convnet import Convnet


class Classifier(ModelPlugin):
    '''Basic classifier.

    '''

    def build(self, input_shape, n_labels, conv_args=[], fc_args=[(200, True, 0.1, 'ReLU')]):
        '''Build classifier model.

        Args:
            conv_args: Convolutional arguments.
            fc_args: Fully-connected arguments.

        '''

        fc_args.append((n_labels, False, False, None))
        self.nets.classifier = Convnet(input_shape, conv_args=conv_args, fc_args=fc_args)

    def routine(self, inputs, targets, criterion=nn.CrossEntropyLoss(reduce=False)):
        '''Classifier routine.

        Args:
            criterion: Classifier criterion.

        '''
        classifier = self.nets.classifier

        outputs = classifier(inputs)[1]
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        unlabeled = targets.eq(-1).long()
        losses = criterion(outputs, (1 - unlabeled) * targets)
        labeled = 1. - unlabeled.float()
        loss = (losses * labeled).sum() / labeled.sum()

        if labeled.sum() > 0:
            correct = 100. * (labeled * predicted.eq(
                targets.data).float()).cpu().sum() / labeled.cpu().sum()
            self.results.accuracy = correct
            self.losses.classifier = loss

        self.results.perc_labeled = labeled.mean()

    def predict(self, inputs):
        classifier = self.nets.classifier

        outputs = classifier(inputs)[1]
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        return predicted

    def visualize(self, images, inputs, targets):
        predicted = self.predict(inputs)
        self.add_image(images.data, labels=(targets.data, predicted.data),
                       name=self.name + '_gt_pred')