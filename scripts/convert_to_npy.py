# Script to convert and resize training and test images into npy

from __future__ import division, print_function

import argparse
from glob import glob
from time import time

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from skimage.io import imread_collection
from skimage.transform import resize

from plankton import utils

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='data/train/')
parser.add_argument('--test_dir', default='data/test/')
parser.add_argument('--submission_sample', default='data/sampleSubmission.csv')
parser.add_argument('--out_dir', default='data/')
parser.add_argument('--hw', default=48, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    print('Reading training images from %s' % args.train_dir)
    dfs = []
    for dir_path in glob('%s*' % args.train_dir):
        label_name = dir_path.split('/')[-1]
        image_path = glob(dir_path + '/*.jpg')
        df_ = pd.DataFrame({'image_path': image_path, 'label_name': label_name})
        dfs.append(df_)
    train_df = pd.concat(dfs, ignore_index=True)
    print('Found %i training images' % len(train_df))

    print('Reading test images from %s' % args.test_dir)
    submission_sample_df = pd.read_csv(args.submission_sample)
    image_path = ['%s%s' % (args.test_dir, fname) for fname in submission_sample_df['image'].values]
    fname = submission_sample_df['image'].values
    print('Found %i test images' % len(submission_sample_df))

    print('Resizing training images to %ix%i...' % (args.hw, args.hw))
    t0 = time()
    imgcol = imread_collection(train_df['image_path'], conserve_memory=True)
    X_train = np.array([resize(img, [args.hw, args.hw]) for img in imgcol])
    X_train = 1 - X_train
    X_train = X_train.astype(np.float32)
    X_train = X_train.reshape(-1, 1, args.hw, args.hw)

    y_enc = LabelEncoder()
    y_train = train_df['label_name']
    y_train = y_enc.fit_transform(y_train).astype(np.int32)

    X_train_train, X_train_test, y_train_train, y_train_test = utils.straified_train_test_split(X_train, y_train, y=y_train, random_state=42, test_size=0.1)

    superclass = map(lambda str: str.split('_')[0], y_enc.classes_)
    y_train_superclass = map(lambda x: superclass[x], y_train)
    y_train_train_superclass = map(lambda x: superclass[x], y_train_train)
    y_train_test_superclass = map(lambda x: superclass[x], y_train_test)

    y_superclass_enc = LabelEncoder().fit(y_train_superclass)
    y_train_train_superclass = y_superclass_enc.fit_transform(y_train_train_superclass).astype(np.int32)
    y_train_test_superclass = y_superclass_enc.fit_transform(y_train_test_superclass).astype(np.int32)

    print('Took %i seconds' % (time() - t0))

    X_train_train_fname = '%sX_train_train_%s' % (args.out_dir, args.hw)
    X_train_test_fname = '%sX_train_test_%s' % (args.out_dir, args.hw)
    y_train_train_fname = '%sy_train_train_%s' % (args.out_dir, args.hw)
    y_train_test_fname = '%sy_train_test_%s' % (args.out_dir, args.hw)
    y_train_train_superclass_fname = '%sy_train_train_superclass_%s' % (args.out_dir, args.hw)
    y_train_test_superclass_fname = '%sy_train_test_superclass_%s' % (args.out_dir, args.hw)
    np.save(X_train_train_fname, X_train_train)
    print('Saved to %s' % X_train_train_fname)
    np.save(X_train_test_fname, X_train_test)
    print('Saved to %s' % X_train_test_fname)
    np.save(y_train_train_fname, y_train_train)
    print('Saved to %s' % y_train_train_fname)
    np.save(y_train_test_fname, y_train_test)
    print('Saved to %s' % y_train_test_fname)
    np.save(y_train_train_superclass_fname, y_train_train_superclass)
    print('Saved to %s' % y_train_train_superclass_fname)
    np.save(y_train_test_superclass_fname, y_train_test_superclass)
    print('Saved to %s' % y_train_test_superclass_fname)

    y_enc_fname = utils.save_to_pickle(y_enc, '%sy_train_encoder' % args.out_dir)
    print('Saved y_train encoder to %s' % y_enc_fname)
    y_superclass_enc_fname = utils.save_to_pickle(y_superclass_enc, '%sy_train_superclass_encoder' % args.out_dir)
    print('Saved y_train_superclass encoder to %s' % y_superclass_enc_fname)

    print('Resizing test images to %ix%i...' % (args.hw, args.hw))
    t0 = time()
    imgcol = imread_collection(image_path, conserve_memory=True)
    X_test = np.array([resize(img, [args.hw, args.hw]) for img in imgcol])
    X_test = 1 - X_test
    X_test = X_test.astype(np.float32)
    X_test = X_test.reshape(-1, 1, args.hw, args.hw)
    print('Took %i seconds' % (time() - t0))

    X_test_fname = '%sX_test_%s' % (args.out_dir, args.hw)
    X_test_fname_fname = '%sX_test_fname_%s' % (args.out_dir, args.hw)
    np.save(X_test_fname, X_test)
    np.save(X_test_fname_fname, fname)
    print('Saved to %s, %s' % (X_test_fname, X_test_fname_fname))

