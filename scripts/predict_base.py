# Script to convert and resize training and test images into npy

from __future__ import division, print_function

import os
import argparse
from time import time, strftime

import pandas as pd
import numpy as np

from plankton import utils

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--X_test_npy', default='data/X_test_48.npy')
parser.add_argument('--X_train_npy', default='data/X_train_48.npy')
parser.add_argument('--X_test_fname_npy', default='data/X_test_fname_48.npy')
parser.add_argument('--enc', default='data/y_train_encoder-2015-03-03-21-01-52.pickle')
parser.add_argument('--submission_sample', default='data/sampleSubmission.csv')
parser.add_argument('--out_dir', default='submissions/')
args = parser.parse_args()

if __name__ == '__main__':
    print('Loading test images from %s and %s' % (args.X_test_npy, args.X_test_fname_npy))
    X = np.load(args.X_test_npy)
    X_train = np.load(args.X_train_npy)
    fname = np.load(args.X_test_fname_npy)

    # Subtract test images from mean of training images
    # X -= np.mean(X_train, axis=0)

    print('Loading model from %s' % args.model)
    net = utils.load_from_pickle(args.model)
    enc = utils.load_from_pickle(args.enc)

    print('Predicting...')
    t0 = time()
    predictions = net.predict_proba(X)
    column_name = np.hstack([['image'], enc.classes_])
    submmission = pd.DataFrame(np.c_[fname, predictions], columns=column_name)
    print('Finished predicting %i images. Took %i seconds' % (len(submmission), time() - t0))

    t0 = time()
    suffix = strftime('%Y-%m-%d-%H-%M-%S')
    csv_fname = '%s%s.csv' % (args.out_dir, suffix)
    print('Saving prediction to %s' % csv_fname)
    submmission.to_csv(csv_fname, index=False, header=True)
    print('Done. Took %i seconds' % (time() - t0))

    print('Ziping up submission...')
    t0 = time()
    zip_fname = '%s%s.zip' % (args.out_dir, suffix)
    zip_cmd = 'zip %s %s' % (zip_fname, csv_fname)
    os.system(zip_cmd)
    print('Zip file created at %s. Took %i seconds' % (zip_fname, time() - t0))
