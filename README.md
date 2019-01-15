# Deep InfoMax (DIM)

[UPDATE]: this work has been accepted as an oral presentation at ICLR 2019. 
We are gradually updating the repository over the next few weeks to reflect experiments in the camera-ready version.

Learning Deep Representations by Mutual Information Estimation and Maximization
Sample code to do the local-only objective in 
https://openreview.net/forum?id=Bklr3j0cKX
https://arxiv.org/abs/1808.06670

### Completed
[Updated 1/15/2019]
* Latest code for dot-product style scoring function for local DIM (single or multiple globals).
* JSD / NCE / DV losses (In addition, f-divergences: KL, reverse KL, squared Hellinger, chi squared).
* Convnet and folded convnet (strided crops) architectures. 

### TODO
* Resnet and folded resnet architectures and training classifiers keeping the encoder fixed (evaluation).
* NDM, MINE, SVM, and MS-SSIM evaluation.
* Global DIM and prior matching.
* Coordinate and occlusion tasks.
* Other baselines (VAE, AAE, BiGAN, NAT, CPC).
* Add nearest neighbor analysis.

### Installation / requirements

This is a package, so to install just run:

    $ pip install .

This package installs the dev branch cortex: https://github.com/rdevon/cortex
Which requires Python 3.5+ (Not tested on higher than 3.7). Note that cortex is in early beta stages, but it is usable for this demo. 

cortex optionally requires visdom: https://github.com/pytorch/vision

You will need to do:

    $ cortex setup

See the cortex README for more info or email us (or submit an issue for legitimate bugs).

### Usage

To get the full set of commands, try:

    $ python scripts/deep_infomax.py --help

For CIFAR10 on a DCGAN architecture, try:

    $ python scripts/deep_infomax.py --d.source CIFAR10 -n DIM_CIFAR10 --d.copy_to_local --t.epochs 1000
    
You should get over 71-72% in the pretraining step alone (this was included for monitoring purposes only). 
Note, this wont get you all the way towards reproducing results in the paper: for this the classifier needs to be retrained with the encoder held fixed.
Support for training a classifier with the representations fixed is coming soon.
    
For STL-10 on folded 64x64 Alexnet (strided crops) with multiple globals and the noise-contrastive estimation type loss, try:

    $ python scripts/deep_infomax.py --d.sources STL10 --d.data_args "dict(stl_resize_only=True)" --d.n_workers 32 -n DIM_STL --t.epochs 200 --d.copy_to_local --encoder_config foldmultialex64x64 --mode nce --global_units 0

### Deep Infomax

TODO: visual guide to Deep Infomax.
