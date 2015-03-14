# Script to train base model
from __future__ import division, print_function

import argparse
from time import time

import numpy as np

from plankton import utils
from plankton import get_net

parser = argparse.ArgumentParser()
parser.add_argument('net_name')
parser.add_argument('--X_train_train_npy', default='data/X_train_train')
parser.add_argument('--X_train_test_npy', default='data/X_train_test')
parser.add_argument('--y_train_train_superclass', default='data/y_train_train_superclass')
parser.add_argument('--y_train_test_superclass', default='data/y_train_test_superclass')
parser.add_argument('--hw', default=48, type=int)
parser.add_argument('--out_dir', default='models/')
args = parser.parse_args()

if __name__ == '__main__':
    print('Loading training images')
    X_train, X_test = np.load('%s_%i.npy' % (args.X_train_train_npy, args.hw)), np.load('%s_%i.npy' % (args.X_train_test_npy, args.hw))
    y_train, y_test = np.load('%s_%i.npy' % (args.y_train_train_superclass, args.hw)), np.load('%s_%i.npy' % (args.y_train_test_superclass, args.hw))
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print('Loading model definition from %s' % args.net_name)
    net = get_net(args.net_name)

    # Pass mean value of training data to the batch iterator
    X_train_mean = np.mean(X_train, axis=0)
    net.batch_iterator_train.mean = X_train_mean
    net.batch_iterator_test.mean = X_train_mean

    t0 = time()
    print('Started training at %s' % t0)
    net.fit(X_train, y_train)
    print('Finished training. Took %i seconds' % (time() - t0))

    y_test_pred = net.predict(X_test)
    y_test_pred_proba = net.predict_proba(X_test)
    lscore = utils.multiclass_log_loss(y_test, y_test_pred_proba)
    ascore = net.score(X_test, y_test)

    print('Accuracy test score is %.4f' % ascore)
    print('Multiclass log loss test score is %.4f' % lscore)

    model_fname = utils.save_to_pickle(net, '%snet-%s-%s' % (args.out_dir, args.net_name, lscore))
    print('Model saved to %s' % model_fname)
