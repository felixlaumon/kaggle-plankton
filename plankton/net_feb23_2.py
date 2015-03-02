"""
Feb 23 4th conv layer with no maxpool at last 2 conv and add translate augmentation
training crashed...
   706  |    0.863500  |    0.874339  |     0.987604  |     73.92%  |  94.6s
best train = 0.852693
best valid = 0.851069 74.66%
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


class TrainBatchIterator(MeanSubtractMixin,
                         ScaleBatchIteratorMixin,
                         RotateBatchIteratorMixin,
                         TranslateBatchIteratorMixin,
                         VerticalFlipBatchIteratorMixin,
                         HorizontalFlipBatchIteratorMixin,
                         ShuffleBatchIteratorMixin,
                         BaseBatchIterator):
    pass


# TODO the mean subtractor should have the mean subtracted from the training
# set instead
class TestBatchIterator(MeanSubtractMixin,
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

        ('l4c', Conv2DCCLayer),
        ('l4d', layers.DropoutLayer),

        ('l5f', layers.DenseLayer),
        ('l5p', layers.FeaturePoolLayer),
        ('l5d', layers.DropoutLayer),

        ('l6f', layers.DenseLayer),
        ('l6p', layers.FeaturePoolLayer),
        ('l6d', layers.DropoutLayer),

        ('l7f', layers.DenseLayer),
        ('l7p', layers.FeaturePoolLayer),
        ('l7d', layers.DropoutLayer),

        ('output', layers.DenseLayer),
    ],

    input_shape=(None, 1, 48, 48),

    l1c_num_filters=64, l1c_filter_size=(3, 3),
    l1p_ds=(2, 2),
    l1d_p=0.2,

    l2c_num_filters=128, l2c_filter_size=(3, 3),
    l2p_ds=(2, 2),
    l2d_p=0.3,

    l3c_num_filters=256, l3c_filter_size=(3, 3),
    l3d_p=0.4,

    l4c_num_filters=512, l4c_filter_size=(3, 3),
    l4d_p=0.4,

    l5f_num_units=2048,
    l5p_ds=2,
    l5d_p=0.5,

    l6f_num_units=2048,
    l6p_ds=2,
    l6d_p=0.5,

    l7f_num_units=2048,
    l7p_ds=2,
    l7d_p=0.5,

    output_num_units=121,
    output_nonlinearity=softmax,

    loss=multinomial_nll,

    batch_iterator_train=TrainBatchIterator(batch_size=128),
    batch_iterator_test=TestBatchIterator(batch_size=128),

    update=rmsprop,
    update_learning_rate=theano.shared(float32(5e-5)),
    update_rho=0.9,
    update_epsilon=1e-6,

    regression=False,

    on_epoch_finished=[
        StepDecay('update_learning_rate', start=5e-5, stop=1e-7),
        EarlyStopping(patience=100)
    ],

    max_epochs=1500,
    verbose=1
)
