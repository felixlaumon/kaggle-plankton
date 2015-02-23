"""
Feb 23 more filters and stronger dropout
   637  |    0.703329  |    0.833093  |     0.844238  |     75.57%  |  76.1s
Early stopping.
Best valid loss was 0.822885 at epoch 536.
Finished training. Took 50076 seconds
Accuracy test score is 0.7680
Multiclass log loss test score is 0.8015
Model saved to models/net-0.801522148043-2015-02-23-16-52-27.pickle
Saving prediction to submissions/2015-02-23-19-44-55.csv
Done. Took 19 seconds
Ziping up submission...
  adding: submissions/2015-02-23-19-44-55.csv (deflated 56%)
Zip file created at submissions/2015-02-23-19-44-55.zip. Took 27 seconds
80  ↑52 felixlaumon
0.822469
19  Mon, 23 Feb 2015 11:48:15
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
                         VerticalFlipBatchIteratorMixin,
                         HorizontalFlipBatchIteratorMixin,
                         ShuffleBatchIteratorMixin,
                         BaseBatchIterator):
    pass


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
        ('l3p', MaxPool2DCCLayer),
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

    l1c_num_filters=64, l1c_filter_size=(3, 3),
    l1p_ds=(2, 2),
    l1d_p=0.2,

    l2c_num_filters=192, l2c_filter_size=(3, 3),
    l2p_ds=(2, 2),
    l2d_p=0.3,

    l3c_num_filters=384, l3c_filter_size=(3, 3),
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

    batch_iterator_train=TrainBatchIterator(batch_size=128),
    batch_iterator_test=TestBatchIterator(batch_size=128),

    update=rmsprop,
    update_learning_rate=theano.shared(float32(1e-4)),
    update_rho=0.9,
    update_epsilon=1e-6,

    regression=False,

    on_epoch_finished=[
        StepDecay('update_learning_rate', start=1e-4, stop=1e-7),
        EarlyStopping(patience=100)
    ],

    max_epochs=1000,
    verbose=1
)
