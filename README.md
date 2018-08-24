# DIM
Learning Deep Representations by Mutual Information Estimation and Maximization

Sample code to do the local-only objective in https://arxiv.org/abs/1808.06670

### Requirements

This module requires cortex: https://github.com/rdevon/cortex

Which requires Python 3.5+ (Not tested on higher than 3.7). Note that cortex is in early beta stages, but it is usable for this demo. 

cortex optionally requires visdom: https://github.com/pytorch/vision

You will need to do:

    $ cortex setup

See the cortex README for more info or email us (or submit an issue for legitimate bugs).

### Usage

To get the full set of commands, try:

    $ python DIM.py --help

For CIFAR10 on a DCGAN architecture, try:

    $ python DIM.py --d.source CIFAR10 -n DIM_CIFAR10 --d.copy_to_local
    
You should get 71-72% in the pretraining step alone (this was included for monitoring purposes only). Retraining the classifiers is not supported in this module, and this is left as an exercise for the reader.
    
For STL-10 on 64x64 Alexnet, try:

    $ python DIM.py --d.source STL10 --encoder_type alexnet -n DIM_STL10 --d.copy_to_local --t.epochs 100
    
If you want to train fully semi-supervised (training the encoder as a classifier), try:

    $ python DIM.py --d.source STL10 --encoder_type alexnet -n DIM_STL10_semi --d.copy_to_local --t.epochs 100 --conv_idx 2 --semi_supervised
