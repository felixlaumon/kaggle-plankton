"""
Feb 21. Batch norm
"""

import theano

from lasagne import layers
from lasagne.layers.cuda_convnet import Conv2DCCLayer, MaxPool2DCCLayer
from lasagne.updates import rmsprop
from lasagne.nonlinearities import softmax
from lasagne.objectives import multinomial_nll
from augment_iterators import *
from net_utils import *
from batchnorm import BatchNormLayer, NeuralNet
# from batchnorm2 import BatchNormLayer, NeuralNet


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

        ('l1c', layers.Conv2DLayer),
        # ('l1b', BatchNormLayer),
        ('l1p', layers.MaxPool2DLayer),
        # ('l1d', layers.DropoutLayer),

        ('l2c', layers.Conv2DLayer),
        # ('l2b', BatchNormLayer),
        ('l2p', layers.MaxPool2DLayer),
        # ('l2d', layers.DropoutLayer),

        ('l3c', layers.Conv2DLayer),
        ('l1b', BatchNormLayer),
        # ('l3d', layers.DropoutLayer),

        ('l5f', layers.DenseLayer),
        ('l5b', BatchNormLayer),
        # ('l5d', layers.DropoutLayer),

        ('l6f', layers.DenseLayer),
        ('l6b', BatchNormLayer),
        # ('l6d', layers.DropoutLayer),

        ('output', layers.DenseLayer),
    ],

    input_shape=(None, 1, 48, 48),

    l1c_num_filters=32, l1c_filter_size=(5, 5), l1c_nonlinearity=None,
    l1p_ds=(2, 2),
    # l1d_p=0.1,

    l2c_num_filters=64, l2c_filter_size=(3, 3), l2c_nonlinearity=None,
    l2p_ds=(2, 2),
    # l2d_p=0.1,

    l3c_num_filters=128, l3c_filter_size=(3, 3), l3c_nonlinearity=None,
    # l3d_p=0.2,

    l5f_num_units=1024, l5f_nonlinearity=None,
    # l5d_p=0.5,

    l6f_num_units=1024, l6f_nonlinearity=None,
    # l6d_p=0.5,

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
