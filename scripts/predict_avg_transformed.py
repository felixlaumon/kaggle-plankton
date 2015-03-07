# Script to convert and resize training and test images into npy

from __future__ import division, print_function

import os
import argparse
import itertools
from time import time
from time import strftime

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from plankton import utils
from plankton.augment_iterators import im_affine_transform

parser = argparse.ArgumentParser()
parser.add_argument('model')

parser.add_argument('--X_test_npy', default='data/X_test_48.npy')
parser.add_argument('--X_train_npy', default='data/X_train_48.npy')
parser.add_argument('--y_train_npy', default='data/X_train_48.npy')
parser.add_argument('--hw', default=48, type=int)
parser.add_argument('--eval_on_train', action='store_true')

parser.add_argument('--X_test_fname_npy', default='data/X_test_fname_48.npy')
parser.add_argument('--enc', default='data/y_train_encoder-2015-02-21-21-04-01.pickle')
parser.add_argument('--submission_sample', default='data/sampleSubmission.csv')
parser.add_argument('--out_dir', default='submissions/')

parser.add_argument('--scale_lower', default=0.85, type=float)
parser.add_argument('--scale_upper', default=1.15, type=float)
parser.add_argument('--scale_step', default=3, type=float)

parser.add_argument('--rotation_lower', default=0, type=int)
parser.add_argument('--rotation_upper', default=360, type=int)
parser.add_argument('--rotation_step', default=36, type=int)

parser.add_argument('--translate_y_lower', default=-2, type=int)
parser.add_argument('--translate_y_upper', default=3, type=int)
parser.add_argument('--translate_y_step', default=2, type=int)

parser.add_argument('--translate_x_lower', default=-2, type=int)
parser.add_argument('--translate_x_upper', default=3, type=int)
parser.add_argument('--translate_x_step', default=2, type=int)

args = parser.parse_args()


def generate_predictions_with_augmented_images(net, X, combinations):
    t0 = time()
    n = X.shape[0]
    num_features = 121
    num_combination = len(combinations)
    predictions = np.empty((n, num_features))
    # TODO optimize by vectorizing the for loop
    # (although X_aug will be very very big and probably won't fit in memory?)
    for i, x in enumerate(X):
        if i % 1000 == 0:
            print('Finished predicting %i images. Took %i seconds so far' % (i, time() - t0))

        # Image only has one channel (grayscale)
        # im_affine_transform only works for images with shape of (height, width)
        x = x[0]
        x_aug = np.array([
            im_affine_transform(x, scale=scale, rotation=rotation, translate_y=translate_y, translate_x=translate_x)
            for scale, rotation, translate_y, translate_x in combinations
        ])

        x_aug = x_aug.reshape(num_combination, 1, args.hw, args.hw).astype(np.float32)
        # TODO do we need to normalize this back to 0?
        pred = net.predict_proba(x_aug).mean(axis=0)
        predictions[i] = pred
    return predictions

if __name__ == '__main__':
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

    print('Loading test images from %s and %s' % (args.X_test_npy, args.X_test_fname_npy))
    X = np.load(args.X_test_npy)
    X_train = np.load(args.X_train_npy)
    fname = np.load(args.X_test_fname_npy)
    hw = args.hw

    print('Loading model from %s' % args.model)
    net = utils.load_from_pickle(args.model)
    enc = utils.load_from_pickle(args.enc)
    # TODO change batchsize to the num_combinaions and see if faster?

    if args.eval_on_train:
        print('Evaluating on training set')
        X_train = np.load(args.X_train_npy)
        y_train = np.load(args.y_train_npy)
        pred_proba_aug = generate_predictions_with_augmented_images(net, X_train, combinations)
        pred_aug = pred_proba_aug.argmax(axis=0)
        pred_prob = net.predict_proba(X_train)
        pred = net.predict(X_train)

        ascore = accuracy_score(y_train, pred)
        lscore = utils.multiclass_log_loss(y_train, pred_prob)
        print('Non-augmented score on training set: ascore=%.4f, lscore=%.4f' % (ascore, lscore))

        ascore_aug = accuracy_score(y_train, pred_aug)
        lscore_aug = utils.multiclass_log_loss(y_train, pred_proba_aug)
        print('Augmented score of training set: ascore=%.4f, lscore=%.4f' % (ascore_aug, lscore_aug))

    print('Predicting...')
    t0 = time()
    predictions = generate_predictions_with_augmented_images(net, X, combinations)
    column_name = np.hstack([['image'], enc.classes_])
    submmission = pd.DataFrame(np.c_[fname, predictions], columns=column_name)
    print('Finished predicting all %i images. Took %i seconds in total' % (len(submmission), time() - t0))

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
