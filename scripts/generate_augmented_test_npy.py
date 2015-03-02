# Script to generate rotated version of the testing set for averaging
# This generate a gigantic file. Probably better off doing augmentation at
# prediction time

from __future__ import division, print_function

import argparse
import itertools
from time import time

import numpy as np

from plankton.augment_iterators import im_affine_transform

parser = argparse.ArgumentParser()
parser.add_argument('--X_test_npy', default='data/X_test_48.npy')
parser.add_argument('--hw', default=48, type=int)

parser.add_argument('--X_test_aug_out_name', default='data/X_test_aug_48.npy')

parser.add_argument('--scale_lower', default=0.9, type=float)
parser.add_argument('--scale_upper', default=1.1, type=float)
parser.add_argument('--scale_step', default=3, type=float)

parser.add_argument('--rotation_lower', default=0, type=int)
parser.add_argument('--rotation_upper', default=360, type=int)
parser.add_argument('--rotation_step', default=45, type=int)

parser.add_argument('--translate_y_lower', default=0, type=int)
parser.add_argument('--translate_y_upper', default=1, type=int)
parser.add_argument('--translate_y_step', default=1, type=int)

parser.add_argument('--translate_x_lower', default=0, type=int)
parser.add_argument('--translate_x_upper', default=1, type=int)
parser.add_argument('--translate_x_step', default=1, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    print('Loading test images')
    X = np.load(args.X_test_npy)
    X = X[:5]
    hw = args.hw

    scale_choices = np.linspace(args.scale_lower, args.scale_upper, args.scale_step)
    rotation_choices = range(args.rotation_lower, args.rotation_upper, args.rotation_step)
    translate_y_choices = range(args.translate_y_lower, args.translate_y_upper, args.translate_y_step)
    translate_x_choices = range(args.translate_x_lower, args.translate_x_upper, args.translate_x_step)

    combinations = list(itertools.product(scale_choices, rotation_choices, translate_y_choices, translate_x_choices))
    num_combination = len(combinations)
    print('scale_choices =', scale_choices)
    print('rotation_choices =', rotation_choices)
    print('translate_y_choices =', translate_y_choices)
    print('translate_x_choices =', translate_x_choices)
    print('Number of augmented images per test image is %i' % num_combination)

    X_aug = np.memmap(args.X_test_aug_out_name, dtype=np.float32, mode='w+', shape=(X.shape[0], num_combination, 1, hw, hw))

    t0 = time()
    for i, x in enumerate(X):
        if i % 2 == 0:
            print('Finished predicting %i images. Took %i seconds so far' % (i, time() - t0))

        # Image only has one channel (grayscale)
        # im_affine_transform only works for images with shape of (height, width)
        x = x[0]
        x_aug = np.array([
            im_affine_transform(x, scale=scale, rotation=rotation, translate_y=translate_y, translate_x=translate_x)
            for scale, rotation, translate_y, translate_x in combinations
        ])
        x_aug = x_aug.reshape(num_combination, 1, hw, hw).astype(np.float32)
        X_aug[i] = x_aug

    X_aug.flush()
    print(X_aug.shape)
    print('All done. Took %i seconds' % (time() - t0))
