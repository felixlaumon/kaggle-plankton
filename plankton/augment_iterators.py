"""Real-time augmentation iterators
Inspired by http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#data-augmentation
"""

import numpy as np
from nolearn.lasagne import BatchIterator as BaseBatchIterator
from skimage.transform import rotate, rescale


class ShuffleBatchIteratorMixin(object):
    """From https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381"""
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShuffleBatchIteratorMixin, self).__iter__():
            yield res


class VerticalFlipBatchIteratorMixin(object):
    """Vertically flip 1/2 of the images at random (uniform)
    """
    def transform(self, Xb, yb):
        Xb, yb = super(VerticalFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        # Select 1/4 images
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb


class HorizontalFlipBatchIteratorMixin(object):
    """Horizontally flip 1/2 of the images at random (uniform)
    """
    def transform(self, Xb, yb):
        Xb, yb = super(HorizontalFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        # Select 1/4 images
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]

        return Xb, yb


class RotateBatchIteratorMixin(object):
    """Rotate 1/2 images at random
    """
    def transform(self, Xb, yb):
        Xb, yb = super(RotateBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        # angle_choices = range(0, 360, 45)
        # angle_choices = range(0, 360, 90)
        angle_choices = range(0, 360, 10)
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = np.array([[im_rotate(img, np.random.choice(angle_choices)) for img in c] for c in Xb[indices]])

        return Xb, yb


class ScaleBatchIteratorMixin(object):
    """Scale for 1/2 images by 0.9 - 1.1 at random
    """
    def transform(self, Xb, yb):
        Xb, yb = super(ScaleBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        scale_choices = np.linspace(.9, 1.1, 10)
        Xb[indices] = np.array([[im_rescale(img, np.random.choice(scale_choices)) for img in c] for c in Xb[indices]])

        return Xb, yb


class TranslateBatchIteratorMixin(object):
    """Translate 1/2 images by -5 to 5 pixels at random
    """
    def transform(self, Xb, yb):
        Xb, yb = super(TranslateBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        translate_choices = range(-5, 6)
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = np.array([[im_translate(img, np.random.choice(translate_choices), np.random.choice(translate_choices)) for img in c] for c in Xb[indices]])

        return Xb, yb


class MeanSubtractMixin(object):
    """Subtract images from mean
    """
    def transform(self, Xb, yb):
        Xb, yb = super(MeanSubtractMixin, self).transform(Xb, yb)
        # TODO should cache mean_X
        self.mean_X = np.mean(self.X, axis=0)
        Xb, yb = np.copy(Xb), np.copy(yb)
        Xb = Xb - self.mean_X
        return Xb, yb


def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


def im_rotate(img, angle, mean=None):
    img = np.copy(img)
    if mean is not None:
        img += mean

    rotated_img = rotate(img, angle, mode='constant', cval=0)

    if mean is not None:
        rotated_img -= mean

    return rotated_img


def im_translate(img, shift_x, shift_y, mean=None):
    img = np.copy(img)
    if mean is not None:
        img += mean

    translate_img = np.zeros_like(img, dtype=img.dtype)

    if shift_x >= 0:
        slice_x_src = slice(None, img.shape[0] - shift_x, None)
        slice_x_tgt = slice(shift_x, None, None)
    else:
        slice_x_src = slice(- shift_x, None, None)
        slice_x_tgt = slice(None, img.shape[0] + shift_x, None)

    if shift_y >= 0:
        slice_y_src = slice(None, img.shape[1] - shift_y, None)
        slice_y_tgt = slice(shift_y, None, None)
    else:
        slice_y_src = slice(- shift_y, None, None)
        slice_y_tgt = slice(None, img.shape[1] + shift_y, None)

    translate_img[slice_x_tgt, slice_y_tgt] = img[slice_x_src, slice_y_src]

    if mean is not None:
        translate_img -= mean

    return translate_img


def im_rescale(img, scale_factor, mean=None):
    img = np.copy(img)
    zoomed_img = np.zeros_like(img, dtype=img.dtype)

    if mean is not None:
        img += mean

    zoomed = rescale(img, scale_factor, mode='constant', cval=0)

    if scale_factor >= 1.0:
        shift_x = (zoomed.shape[0] - img.shape[0]) // 2
        shift_y = (zoomed.shape[1] - img.shape[1]) // 2
        zoomed_img[:, :] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
    else:
        shift_x = (img.shape[0] - zoomed.shape[0]) // 2
        shift_y = (img.shape[1] - zoomed.shape[1]) // 2
        zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

    if mean is not None:
        zoomed_img -= mean

    return zoomed_img
