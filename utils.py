"""Misc utilities"""

import sys
sys.setrecursionlimit(10000000)

from time import strftime
import cPickle as pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


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
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()


def write_submission_csv(net):
    X_test, _ = load_data(test=True)
    y_pred = net.predict(X_test)
    image_id = np.linspace(1, len(y_pred), len(y_pred)).astype(np.int32)

    predictions = []
    for i in range(len(y_pred)):
        predictions.append((image_id[i], y_pred[i]))

    submission = pd.DataFrame(predictions, columns=['ImageId', 'Label'])
    now_str = datetime.now().isoformat().replace(':', '-')
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)


def save(x, filename):
    suffix = strftime('%Y-%m-%d-%H-%M-%S')
    with open('%s-%s.pickle' % (filename, suffix), 'wb') as f:
        pickle.dump(x, f, -1)


def load(filename):
    return pickle.load(open(filename, 'rb'))
