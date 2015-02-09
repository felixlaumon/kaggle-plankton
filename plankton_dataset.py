import numpy as np

from pylearn2.datasets import DenseDesignMatrix
from sklearn.preprocessing import LabelBinarizer


class PlanktonDataset(DenseDesignMatrix):

    def __init__(self, which_set='train'):
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_test = np.load('data/X_test.npy')

        if which_set == 'train':
            X = X_train
            enc = LabelBinarizer()
            y = enc.fit_transform(y_train)
        else:
            X = X_test

        X = X.reshape(-1, 48, 48, 1)

        super(PlanktonDataset, self).__init__(X=X, y=y)
