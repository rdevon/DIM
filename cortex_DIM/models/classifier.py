'''Classification models.

'''

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.plugins import ModelPlugin

from cortex_DIM.nn_modules.convnet import Convnet


class Classifier(ModelPlugin):
    '''Basic classifier.

    '''

    def build(self, input_shape, n_labels,
              layers=[dict(layer='linear', args=(200,), bn=True, do=0.1, act='ReLU')]):
        '''Build classifier model.

        Args:
            layers: list of layer arguments. See `nn_modules.convnet` for details.

        '''
        layers = copy.deepcopy(layers)
        layers.append(dict(layer='linear', args=(n_labels,)))
        self.add_nets(classifier=Convnet(input_shape, layers=layers))

    def routine(self, inputs, targets, criterion=nn.CrossEntropyLoss(reduce=False), top_K=None):
        '''Classifier routine.

        Args:
            criterion: Classifier criterion.
            top_K (int or None): Evaluate top-K accuracy.

        '''
        classifier = self.nets.classifier

        outputs = classifier(inputs)
        unlabeled = targets.eq(-1).long()
        losses = criterion(outputs, (1 - unlabeled) * targets)
        labeled = 1. - unlabeled
        loss = (losses * labeled.float()).sum() / labeled.float().sum()

        if labeled.sum() > 0:
            acc = self.accuracy(inputs, targets, labeled)
            if top_K is not None:
                top_K_acc = self.accuracy(inputs, targets, labeled, top=top_K)
                self.add_results(**{'accuracy_top_{}'.format(top_K): top_K_acc})
            self.add_results(accuracy_top_1=acc)
            self.add_losses(classifier=loss)

    def predict(self, inputs):
        '''Make prediction.

        '''
        with torch.no_grad():
            classifier = self.nets.classifier

            outputs = classifier(inputs)
            predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]
            return predicted

    def accuracy(self, inputs, targets, labeled, top=1):
        '''Computes the accuracy.

        Args:
            inputs: Classifier inputs.
            targets: Targets for each input.
            labeled: Binary variable indicating whether a target exists.
            top (int): Top-K accuracy.

        '''
        with torch.no_grad():
            outputs = self.nets.classifier(inputs)

            _, pred = outputs.topk(top, 1, True, True)
            pred = pred.t()
            correct = labeled.float() * pred.eq(targets.view(1, -1).expand_as(pred)).float()

            correct_k = correct[:top].view(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(100.0 / labeled.float().sum())
            return accuracy.detach().item()

    def visualize(self, images, inputs, targets):
        '''Visualize.

        Shows classification prediction along with ground truth as image.

        '''
        predicted = self.predict(inputs).detach()
        self.add_image(images.data, labels=(targets.data, predicted.data),
                       name=self.name + '_gt_pred')
