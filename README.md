# Deep InfoMax (DIM)

This work has been accepted as an oral presentation at ICLR 2019.
We are gradually updating the repository over the next few weeks to reflect experiments in the camera-ready version.

Learning Deep Representations by Mutual Information Estimation and Maximization
Sample code to do the local-only objective in 
https://openreview.net/forum?id=Bklr3j0cKX
https://arxiv.org/abs/1808.06670

### Completed
[Updated 1/18/2019]
* Latest code for dot-product style scoring function for local DIM (single or multiple globals).
* JSD / NCE / DV losses (In addition, f-divergences: KL, reverse KL, squared Hellinger, chi squared).
* Convnet and folded convnet (strided crops) architectures. 
* Resnet and folded resnet architectures.
* Training classifiers keeping the encoder fixed (evaluation).

### TODO
* NDM, MINE, SVM, and MS-SSIM evaluation.
* Semi-supervised learning.
* Global DIM and prior matching.
* Coordinate and occlusion tasks.
* Other baselines (VAE, AAE, BiGAN, NAT, CPC).
* Add nearest neighbor analysis.

### Installation / requirements

This is a package, so to install just run:

    $ pip install git+https://github.com/rdevon/cortex@dev#egg=cortex-dev0.13
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
    
For classification evaluation, keeping the encoder fixed (the classification numbers for the above script are for monitoring only),
do:

    $ python scripts/evaluation.py --d.source CIFAR10 -n DIM_CIFAR10_cls --d.copy_to_local --t.epochs 1000 -L <path to cortex outs>/DIM_CIFAR10/binaries/DIM_CIFAR10_final.t7
    
You should get 73-74% with this model. 
Note that the learning rate schedule in this script isn't precisely what was used across models in the paper.
The rates in the paper were different for different classifiers; if there is any significant classifier overfitting, adjust to use a faster decay rate.

For a folded Resnet (strided crops) and the noise-contrastive estimation (NCE) type loss, one could do:

    $ python scripts/deep_infomax.py --d.source CIFAR10 --encoder_config foldresnet19_32x32 --mode nce --mi_units 1024 -n DIM_CIFAR10_FoldedResnet --d.copy_to_local --t.epochs 1000
    
where the number of units used for estimating mutual information are the same we used for comparisons to CPC. 
For STL-10 on folded 64x64 Alexnet with multiple globals and the NCE-type loss, try:

    $ python scripts/deep_infomax.py --d.sources STL10 --d.data_args "dict(stl_resize_only=True)" --d.n_workers 32 -n DIM_STL --t.epochs 200 --d.copy_to_local --encoder_config foldmultialex64x64 --mode nce --global_units 0

### Deep Infomax

TODO: visual guide to Deep Infomax.
