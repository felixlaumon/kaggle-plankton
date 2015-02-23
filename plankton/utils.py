"""Misc utilities"""

import sys
from time import strftime
import cPickle as pickle

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.manifold import TSNE

# Allow pickle to work with very deep recursion
sys.setrecursionlimit(10000000)


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Taken from https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/2644/multi-class-log-loss-function/14250#post14250

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


def straified_train_test_split(*arrays, **kwargs):
    """Like train_test_split but stratified
    Note that y is a required keyword argument
    """

    y = kwargs['y']
    test_size = kwargs.pop('test_size', 0.25)
    random_state = kwargs.pop('random_state', None)
    sss = StratifiedShuffleSplit(y, test_size=test_size, random_state=random_state)
    train, test = iter(sss).next()

    return flatten([[a[train], a[test]] for a in arrays])


def flatten(l):
    """http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python"""
    return [item for sublist in l for item in sublist]


def plot_sample(ax, x, y, y_pred=None):
    if y_pred is not None:
        xlabel = 'actual {}, predicted {}'.format(y, y_pred)
    else:
        xlabel = 'actual {}'.format(y)

    ax.set_xlabel(xlabel)
    ax.imshow(x[0], cmap='gray')


def plot_samples(X, y, y_pred=None, amount=25, grid=(5, 5), random=True):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.35, wspace=0.35)

    if random:
        indices = [int(np.random.sample() * X.shape[0]) for x in range(amount)]
    else:
        indices = range(amount)

    for i, index in enumerate(indices):
        ax = fig.add_subplot(grid[0], grid[1], i + 1, xticks=[], yticks=[])
        pred = y_pred[index] if y_pred is not None else None
        plot_sample(ax, X[index], y[index], y_pred=pred)

    plt.show()


def plot_loss(net):
    train_loss = np.array([i['train_loss'] for i in net.train_history_])
    valid_loss = np.array([i['valid_loss'] for i in net.train_history_])
    plt.figure(figsize=(8, 8))
    plt.title('training loss vs. validation loss')
    plt.plot(train_loss, label='training')
    plt.plot(valid_loss, label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()


def save_to_pickle(x, filename):
    suffix = strftime('%Y-%m-%d-%H-%M-%S')
    fname = '%s-%s.pickle' % (filename, suffix)
    with open(fname, 'wb') as f:
        pickle.dump(x, f, -1)
    return fname


def load_from_pickle(filename):
    return pickle.load(open(filename, 'rb'))


def vis_layer_weights(layer, size=None):
    W = layer.W.get_value()
    b = layer.b.get_value()
    f = [w + bb for w, bb in zip(W, b)]
    n = layer.num_filters

    if size is None:
        n = np.ceil(np.sqrt(n)).astype(np.int)
        size = (n, n)

    plt.title(layer.__class__.__name__)
    gs = gridspec.GridSpec(size[0], size[1])
    for i in range(layer.num_filters):
        g = gs[i]
        ax = plt.subplot(g)
        ax.grid()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(f[i][0], interpolation='nearest', cmap='gray')


def vis_layer_output(layer, input, size=None):
    output = layer.get_output(input).eval()[0]
    n = layer.num_filters

    if size is None:
        n = np.ceil(np.sqrt(n)).astype(np.int)
        size = (n, n)

    plt.title(layer.__class__.__name__)
    gs = gridspec.GridSpec(size[0], size[1])
    for i in range(layer.num_filters):
        g = gs[i]
        ax = plt.subplot(g)
        ax.grid()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(output[i], interpolation='nearest', cmap='gray')


def plot_tsne(layer, input, labels):
    output = layer.get_output(input).eval()
    shape = output.shape
    output = output.reshape(shape[0], -1)

    tsne = TSNE(n_components=2, random_state=0)
    # bug in tsne https://github.com/scikit-learn/scikit-learn/issues/4124
    output = np.asfarray(output, dtype='float')
    xx, yy = tsne.fit_transform(output).transpose()

    plt.title(layer.__class__.__name__)
    plt.scatter(yy, xx, c=labels, cmap='jet')


def vis_hidden_layer_weight(layer, input):
    pass
