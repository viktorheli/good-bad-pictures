# good-bad-pictures
Neuralnet filter for good or bad pictures. Crated for personalizing filtering for pictures feed in agakat.com project. 
We create dataset and train neural network to distinguish between bad and good pictures. The neural network will filter the pictures according to the preferences of the one who chose the pictures for the dataset.  

This project consist of several parts:
1. make-my-dataset.lua - create dataset in t7 fromat from pre-selected pictures. Need two categories - GOOD and BAD
This script has two parameters:
-path Path to train or test dataset in flat structure. For example train/bad and train/good
-filename Filename for saving our dataset. For example my-trainset.t7

For example use:
th make-my-dataset.lua -filename for-github.t7 -path dataset/train/


What need for work:

Nvidia CUDA

Torch - http://torch.ch/

And modules:

paths - luarocks install paths

image - luarocks instal image

nn - luarocks install nn

optim - luarocks install optim

cutorch - luarocks install cutorch

cunn - luarocks install cutorch
