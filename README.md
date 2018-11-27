# Convolutional_Deep_Belief_Network
This code contains how to create a convolutional DBN from stacked convolutional RBM, configure it and train it layerwise. 


## Description of the project
A convolutional deep belief network (CDBN) is a deep network which consists in a stack of convolutional restricted boltzmann machine (CRBM). 
Because the gradient of the network is intractable, a greedy layer-wise training procedure is used. 
More details can be found [here](https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf) and [here](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf) and [here](https://www.cs.toronto.edu/~hinton/science.pdf).
This project contains 3 files, `cdbn_backup.py`, `crbm_backup.py` and `demo_cdbn_mnist.py`.
Below is a description of each file, what it does and how to use it.


## Model overview
![figure1](https://i.stack.imgur.com/J7FZG.jpg)


## Requirement
- Python 2.7.6 or 3
- Tensorflow 0.12 (or above) with GPU supported
- Numpy


## How to use
1. crbm_backup.py
This file is the building block of the whole network since it contains the class that is necessary for ONE crbm to function properly. 
Many parameters are included such as dimension of input and hidden, parameter to inialitize hidden unit, whether they are gaussian or not, whether to use probabilistic max pooling, whether to use sparsity, etc.
For one crbm, one can compute its energy, infer the probability forward or backward, draw samples forward or backward, do contrastive divergence. This is the most complicated part of the project since the contrastive divergence does not rely on computation of gradiant but rather make use of Gibbs sampling. 

2. cdbn_backup.py
This file is the class that represent the whole network and can be composed of several crbm that stacked together. First the network is created empty and then layers can be added successively. A final softmax layer can also be added. After locking the network, the whole network can be trained by training each layer successively.

3. demo_cdbn_mnist.py
This file contain a simple working example of CDBN used on the MNIST dataset. 
After training, you can see the accuracy of the network. This accuracy should not be very high since the network is not tuned for this in specific and I did not modify this project for at least half a year. Rather the whole pipeline seems to be working and this is the most important point.

