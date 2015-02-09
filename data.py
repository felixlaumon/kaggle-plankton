"""Load data from /data and other data pre-processing"""
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from glob import glob
from itertools import cycle
from joblib import Parallel, delayed
import multiprocessing

from skimage.io import imread
from skimage.transform import rotate, resize, rescale

NUM_CORES = multiprocessing.cpu_count()
TRAINING_CSV = './data/train.csv'
TESTING_CSV = './data/test.csv'
IMAGE_WIDTH = IMAGE_HEIGHT = 48


def load_data(test=False, scale=True, resize_shape=(IMAGE_WIDTH, IMAGE_HEIGHT), random_n=None):
    """Load images"""
    half_pixel_value = 255 / 2

    if test:
        imagepaths = glob('data/test/*.jpg')
        filenames = [path.split('/')[-1] for path in imagepaths]
        images = [imread(path) for path in imagepaths]
        flattened = zip(filenames, images)
        print('Finished reading directories; found {} images'.format(len(flattened)))

        df = pd.DataFrame(flattened, columns=['filename', 'image'])
        X = df['image'].values
        y = df['filename'].values

    else:
        dir_names = glob('data/train/*')
        labels = [name.replace('data/train/', '') for name in dir_names]
        # labels_i = range(len(labels))
        imagepaths_by_label = [glob(name + '/*.jpg') for name in dir_names]
        filenames_by_label = [[path.split('/')[-1].replace('.jpg', '') for path in paths] for paths in imagepaths_by_label]
        images_by_label = [[imread(path) for path in paths] for paths in imagepaths_by_label]
        combined = zip(labels, imagepaths_by_label, filenames_by_label, images_by_label)
        merged = [zip(cycle([label]), paths, filenames, images) for label, paths, filenames, images in combined]
        flattened = reduce(lambda x, y: x + y, merged)
        print('Finished reading directories; found {} images'.format(len(flattened)))

        df = pd.DataFrame(flattened, columns=['label', 'path', 'filename', 'image'])
        X = df['image'].values
        y = df['label'].values

    if random_n:
        print('Truncating data to {} row'.format(random_n))
        indices = np.random.choice(len(flattened), random_n)
        X = X[indices]
        y = y[indices]

    if scale:
        print('Scaling image output')
        X = (X - half_pixel_value) / half_pixel_value

    if resize_shape:
        print('Resizing all images to {}'.format(resize_shape))
        # result = Parallel(n_jobs=NUM_CORES)(delayed(resize)(x, resize_shape) for x in X)
        # X = np.array(result)
        X = np.array([resize(x, resize_shape) for x in X])
        X = X.reshape(-1, 1, resize_shape[0], resize_shape[1])

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    return X, y


def scale_retain_aspect(x, target_shape):
    """Scale image x to target_shape. Pad with ones if necessary"""
    scale_factor = target_shape[0] / (x.shape[0] if x.shape[0] > x.shape[1] else x.shape[1])
    img = rescale(x, scale_factor)
    ones_width = target_shape[0] - img.shape[0]
    ones_height = target_shape[1] - img.shape[1]

    if ones_width:
        ones = np.ones([ones_width, target_shape[1]])
        img = np.vstack([img, ones])
    else:
        ones = np.ones([target_shape[0], ones_height])
        img = np.hstack([img, ones])

    return img
