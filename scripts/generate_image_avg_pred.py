# Script to convert and resize training and test images into npy

from __future__ import division, print_function

import os
import argparse
import itertools
from time import time, strftime

import pandas as pd
import numpy as np

from plankton import utils
from plankton.augment_iterators import im_affine_transform

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('pred_aug_fname')

parser.add_argument('--X_train_npy', default='data/X_train_48.npy')
parser.add_argument('--hw', default=48, type=int)

parser.add_argument('--scale_lower', default=0.85, type=float)
parser.add_argument('--scale_upper', default=1.15, type=float)
parser.add_argument('--scale_step', default=3, type=float)

parser.add_argument('--rotation_lower', default=0, type=int)
parser.add_argument('--rotation_upper', default=360, type=int)
parser.add_argument('--rotation_step', default=90, type=int)

parser.add_argument('--translate_y_lower', default=2, type=int)
parser.add_argument('--translate_y_upper', default=-2, type=int)
parser.add_argument('--translate_y_step', default=3, type=int)

parser.add_argument('--translate_x_lower', default=2, type=int)
parser.add_argument('--translate_x_upper', default=-2, type=int)
parser.add_argument('--translate_x_step', default=3, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    print('Loading test images from %s' % (args.X_train_npy))
    X = np.load(args.X_train_npy)
    hw = args.hw

    print('Loading model from %s' % args.model)
    net = utils.load_from_pickle(args.model)
    # if 'mean' not in net.batch_iterator_test:
    #     print('Warning: net.batch_iterator_test does not have preset mean value. Using mean from all training data for now')
    net.batch_iterator_test.mean = np.mean(X, axis=0)

    scale_choices = np.linspace(args.scale_lower, args.scale_upper, args.scale_step)
    rotation_choices = range(args.rotation_lower, args.rotation_upper, args.rotation_step)
    translate_y_choices = range(args.translate_y_lower, args.translate_y_upper, args.translate_y_step)
    translate_x_choices = range(args.translate_x_lower, args.translate_x_upper, args.translate_x_step)

    combinations = list(itertools.product(scale_choices, rotation_choices, translate_y_choices, translate_x_choices))
    num_combination = len(combinations)
    print('scale_choices = ', scale_choices)
    print('rotation_choices = ', rotation_choices)
    print('translate_y_choices = ', translate_y_choices)
    print('translate_x_choices = ', translate_x_choices)
    print('Number of augmented images per test image is %i' % num_combination)

    print('Predicting...')
    t0 = time()

    # For each image generate a number of variation of the same images using
    # affine transform and average out prediction
    predictions = np.empty((X.shape[0], num_combination, 121))
    for i, x in enumerate(X):
        if i % 2500 == 0:
            print('Finished predicting %i images. Took %i seconds so far' % (i, time() - t0))

        # Image only has one channel (grayscale)
        # im_affine_transform only works for images with shape of (height, width)
        x = x[0]
        x_aug = np.array([
            im_affine_transform(x, scale=scale, rotation=rotation, translate_y=translate_y, translate_x=translate_x)
            for scale, rotation, translate_y, translate_x in combinations
        ])
        x_aug = x_aug.reshape(num_combination, 1, hw, hw).astype(np.float32)
        pred = net.predict_proba(x_aug)
        predictions[i] = pred
    print('Finished predicting all %i images. Took %i seconds in total' % (X.shape[0], time() - t0))

    np.save(args.pred_aug_fname, predictions)
    print('Predictions saved to %s' % args.pred_aug_fname)
    print(predictions.shape)
