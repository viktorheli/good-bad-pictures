# good-bad-pictures
Neuralnet filter for good or bad pictures. Created for personalizing filtering for pictures feed in https://agakat.com project. 
We create dataset and train neural network to distinguish between bad and good pictures. The neural network will filter the pictures according to the preferences of the one who chose the pictures for the dataset. If you take a large set of pictures from different people, then the filter will be more universal. To simple learning the network enough 100 pictures for each category. 

##This project consist of several scripts:

1. make-my-dataset.lua - create dataset in t7 fromat from pre-selected pictures. Need two categories - GOOD and BAD, also with this script we created test dataset. 
This script has two parameters:

-path Path to train or test dataset in flat structure. For example train/bad and train/good

-filename Filename for saving our dataset. For example my-trainset.t7

For example use:
th make-my-dataset.lua -filename for-github.t7 -path dataset/train/


2. pic-train-test.lua - script to training netwoks and forwarding images through network


This script has following parameters:

-batchsize size of batch for training. Default is 5

-img       path to image for test. Default is none

-save      path to save. Default local directory

-storenet  File name to stote training data. Default is my-network.dat

-clearnet  File name to stote clear net data. It reduce file size, but clear training data. Default my-clear-network.dat 

-train     train iterations 

-trainset  Dataset to train network Default is for-github.t7

-testset   Dataset to train network Default for-github-test.t7

-imgdisp   Display or not display image. Dafult is no. 
How to use:

For training: 

th fotograf-cuda.lua -storenet github.dat -clearnet github-clear.dat -trainset for-github.t7 -testset for-github-test.t7 -train 100

Forwarding any image throgh net:

th pic-train-test.lua -img dataset/test/bad/534fac49dbcedb597a8b4e9e.jpg -clearnet github-clear.dat

In this case we will see probability of categories:

bad     100

good    0

3. If you are use neural network in web projects reasonable strategy may be loading network in memory as a daemon and use it permanently in hidden server throgh web web requests from other computers.   
Script good-bad-daemon.lua - is a daemon for listening tcp port and wait picture from client. In response, the daemon send to the client, probabilty in each categories. As example:

bad: 99.767

good: 0.233


4. Not everyone has a GPU in servers for processing neural networks. :) In this case the better solution may be following: 

Train you model on GPU in you workstation or Amazon cloud 
Convert model from CUDA to CPU.
Use the model in you ouwn server without GPU. 
Profit! :) 

This solution work better in case when workload is not big and time to processing images is not very critical parameter. 

Simple script convert-cuda-cpu.lua help as to convert CUDA model to CPU model.

Script has following key parameters:

  -cudamodel Filename with cuda neuralnet.

  -cpumodel  Filename to save converted cpu model

Example use:

th convert-cuda-cpu.lua -cudamodel github.dat -cpumodel github-cpu.dat
----Loading CUDA model----
Loding completed in: 274ms

----Converting model----
Converting completed in: 171ms

----Saving CPU model----
Saving CPU model completed in: 252ms

All done successfully. CPU model saved to file: github-cpu.dat


Links

Simple dataset with 90 images can bee downloaded here:

https://www.dropbox.com/s/273dg4ecesqhu67/for-github.t7.tar.gz?dl=0

Testset with 10 images can bee downloaded here:

https://www.dropbox.com/s/pbp07g7686cmjwc/for-github-test.t7.tar.gz?dl=0

A bit pretrained model with training data can be downloaded here:

https://www.dropbox.com/s/6jfns1kawqnfxrt/github.dat.tar.gz?dl=0

Clear model without training data can be downloaded here: 

https://www.dropbox.com/s/sqdgk2kfyxb2t0g/github-clear.dat.tar.gz?dl=0




What need for work:

Nvidia CUDA

Torch - http://torch.ch/ or https://github.com/torch

And modules:

paths - https://github.com/torch/paths

image - https://github.com/torch/image

nn - https://github.com/torch/nn

optim - https://github.com/torch/optim

cutorch - https://github.com/torch/cutorch

cunn - https://github.com/torch/cunn

waffle - https://github.com/benglard/waffle

cudnn - https://github.com/soumith/cudnn.torch
