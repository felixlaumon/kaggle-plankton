"""
Feb 26



"""

import theano

from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer
from lasagne.updates import rmsprop
from lasagne.nonlinearities import softmax
from lasagne.objectives import multinomial_nll
from augment_iterators import *
from net_utils import *


class TrainBatchIterator(MeanSubtractMixin,
                         AffineTransformIteratorMixin,
                         VerticalFlipBatchIteratorMixin,
                         HorizontalFlipBatchIteratorMixin,
                         ShuffleBatchIteratorMixin,
                         BaseBatchIterator):
    pass


class TestBatchIterator(MeanSubtractMixin,
                        BaseBatchIterator):
    pass


train_iterator = TrainBatchIterator(batch_size=128,
                                    vflip_n=2, hflip_n=2, affine_n=2,
                                    angle_choices=range(0, 360, 10),
                                    scale_choices=np.linspace(.8, 1.2, 10),
                                    translate_choices=range(-3, 4))
test_iterator = TestBatchIterator(batch_size=128)

net = NeuralNet(
    eval_size=0.05,

    layers=[
        ('input', layers.InputLayer),

        ('l1c', Conv2DDNNLayer),
        ('l1p', MaxPool2DDNNLayer),
        ('l1d', layers.DropoutLayer),

        ('l2c', Conv2DDNNLayer),
        ('l2p', MaxPool2DDNNLayer),
        ('l2d', layers.DropoutLayer),

        ('l3c', Conv2DDNNLayer),
        ('l3p', MaxPool2DDNNLayer),
        ('l3d', layers.DropoutLayer),

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

    l1c_num_filters=128, l1c_filter_size=(3, 3),
    l1p_ds=(2, 2),
    l1d_p=0.2,

    l2c_num_filters=256, l2c_filter_size=(3, 3),
    l2p_ds=(2, 2),
    l2d_p=0.3,

    l3c_num_filters=512, l3c_filter_size=(3, 3),
    l3p_ds=(2, 2),
    l3d_p=0.4,

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

    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    update=rmsprop,
    update_learning_rate=theano.shared(float32(1e-4)),
    update_rho=0.9,
    update_epsilon=1e-6,

    regression=False,

    on_epoch_finished=[
        StepDecay('update_learning_rate', start=1e-4, stop=1e-7),
        EarlyStopping(patience=100)
    ],

    max_epochs=1500,
    verbose=1
)
