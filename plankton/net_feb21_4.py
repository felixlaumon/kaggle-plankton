"""
Feb 21. All regularized
   182  |    0.738898  |    1.131888  |     0.652802  |     71.13%  |  20.0s
Early stopping.
Best valid loss was 1.009449 at epoch 81.
Finished training. Took 3663 seconds
Accuracy test score is 0.7208
Multiclass log loss test score is 0.9982
Model saved to models/net-0.998242698556-2015-02-22-02-59-28.pickle
"""

import theano

from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer
from lasagne.updates import rmsprop
from lasagne.nonlinearities import softmax
from lasagne.objectives import multinomial_nll
from augment_iterators import *
from net_utils import *


class BatchIterator(MeanSubtractMixin,
                    RotateBatchIteratorMixin,
                    VerticalFlipBatchIteratorMixin,
                    HorizontalFlipBatchIteratorMixin,
                    BaseBatchIterator):
    pass

net = NeuralNet(
    eval_size=0.05,

    layers=[
        ('input', layers.InputLayer),

        ('l1c', Conv2DCCLayer),
        ('l1p', MaxPool2DCCLayer),
        ('l1d', layers.DropoutLayer),

        ('l2c', Conv2DCCLayer),
        ('l2p', MaxPool2DCCLayer),
        ('l2d', layers.DropoutLayer),

        ('l3c', Conv2DCCLayer),
        ('l3d', layers.DropoutLayer),

        ('l5f', layers.DenseLayer),
        ('l5d', layers.DropoutLayer),

        ('l6f', layers.DenseLayer),
        ('l6d', layers.DropoutLayer),

        ('output', layers.DenseLayer),
    ],

    input_shape=(None, 1, 48, 48),

    l1c_num_filters=32, l1c_filter_size=(5, 5),
    l1p_ds=(2, 2),
    l1d_p=0.1,

    l2c_num_filters=64, l2c_filter_size=(3, 3),
    l2p_ds=(2, 2),
    l2d_p=0.1,

    l3c_num_filters=128, l3c_filter_size=(3, 3),
    l3d_p=0.2,

    l5f_num_units=1024,
    l5d_p=0.5,

    l6f_num_units=1024,
    l6d_p=0.5,

    output_num_units=121,
    output_nonlinearity=softmax,

    loss=multinomial_nll,

    batch_iterator_train=BatchIterator(batch_size=128),
    batch_iterator_test=BatchIterator(batch_size=128),

    update=rmsprop,
    update_learning_rate=theano.shared(float32(1e-3)),
    update_rho=0.9,
    update_epsilon=1e-6,

    regression=False,

    on_epoch_finished=[
        StepDecay('update_learning_rate', start=1e-3, stop=1e-7),
        EarlyStopping(patience=100)
    ],

    max_epochs=1000,
    verbose=1
)
