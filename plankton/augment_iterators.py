"""Real-time augmentation iterators
Portion of the code taken from:
- http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#data-augmentation
- https://github.com/benanne/kaggle-galaxies/blob/master/realtime_augmentation.py
"""

import numpy as np
from skimage.transform import _warps_cy
from skimage.transform import rotate
from skimage.transform import rescale
from skimage.transform import warp
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform


class BaseBatchIterator(object):
    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb


class ShuffleBatchIteratorMixin(object):
    """From https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381"""
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShuffleBatchIteratorMixin, self).__iter__():
            yield res


class VerticalFlipBatchIteratorMixin(object):
    """Vertically flip images at random (uniform)
    """
    def __init__(self, vflip_n=2, *args, **kwargs):
        self.vflip_n = vflip_n
        super(VerticalFlipBatchIteratorMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(VerticalFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        # Select 1/4 images
        n = Xb.shape[0]
        indices = np.random.choice(n, n / self.vflip_n, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb


class HorizontalFlipBatchIteratorMixin(object):
    """Horizontally flip images at random (uniform)
    """
    def __init__(self, hflip_n=2, *args, **kwargs):
        self.hflip_n = hflip_n
        super(HorizontalFlipBatchIteratorMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(HorizontalFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        n = Xb.shape[0]
        indices = np.random.choice(n, n / self.hflip_n, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]

        return Xb, yb


class RotateBatchIteratorMixin(object):
    """Rotate images at random
    """
    def __init__(self, rotate_n=2, angle_choices=range(0, 360, 10), *args, **kwargs):
        self.rotate_n = rotate_n
        self.angle_choices = angle_choices
        super(RotateBatchIteratorMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(RotateBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        n = Xb.shape[0]
        indices = np.random.choice(n, n / self.rotate_n, replace=False)
        Xb[indices] = np.array([[im_rotate(img, np.random.choice(self.angle_choices)) for img in c] for c in Xb[indices]])

        return Xb, yb


class ScaleBatchIteratorMixin(object):
    """Scale images by 0.9 - 1.1 at random
    """
    def __init__(self, scale_n=2, scale_choices=np.linspace(.9, 1.1, 10), *args, **kwargs):
        self.scale_n = scale_n
        self.scale_choices = scale_choices
        super(ScaleBatchIteratorMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(ScaleBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        n = Xb.shape[0]
        indices = np.random.choice(n, n / self.scale_n, replace=False)
        Xb[indices] = np.array([[im_rescale(img, np.random.choice(self.scale_choices)) for img in c] for c in Xb[indices]])

        return Xb, yb


class TranslateBatchIteratorMixin(object):
    """Translate 1/2 images by -5 to 5 pixels at random
    """
    def __init__(self, translate_n=2, translate_choices=range(-5, 6), *args, **kwargs):
        self.translate_n = translate_n
        self.translate_choices = translate_choices
        super(TranslateBatchIteratorMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(TranslateBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        n = Xb.shape[0]
        indices = np.random.choice(n, n / self.translate_n, replace=False)
        Xb[indices] = np.array([[im_translate(img, np.random.choice(self.translate_choices), np.random.choice(self.translate_choices)) for img in c] for c in Xb[indices]])

        return Xb, yb


class MeanSubtractMixin(object):
    """Subtract images from mean
    """
    def __init__(self, mean=None, *args, **kwargs):
        self.mean = mean
        super(MeanSubtractMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(MeanSubtractMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)
        Xb = Xb - self.mean
        return Xb, yb


class AffineTransformIteratorMixin(object):
    """Apply affine transformation
    """
    def __init__(self, affine_n=2,
                 scale_choices=np.linspace(.9, 1.1, 10),
                 translate_choices=range(-5, 6),
                 angle_choices=range(0, 360, 10), *args, **kwargs):
        self.affine_n = affine_n
        self.scale_choices = scale_choices
        self.translate_choices = translate_choices
        self.angle_choices = angle_choices
        super(AffineTransformIteratorMixin, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        Xb, yb = super(AffineTransformIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)
        n = Xb.shape[0]
        indices = np.random.choice(n, n / self.affine_n, replace=False)
        Xb[indices] = np.array(
            [[im_affine_transform(
                img, scale=np.random.choice(self.scale_choices),
                rotation=np.random.choice(self.angle_choices),
                translate_y=np.random.choice(self.translate_choices),
                translate_x=np.random.choice(self.translate_choices)
                ) for img in c] for c in Xb[indices]])
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


def im_affine_transform(img, scale, rotation, translate_y, translate_x):
    img = np.copy(img)

    # Normalize so that the param acts more like im_rotate, im_translate etc.
    scale = 1 / scale
    translate_x = - translate_x

    # shift to center first so that image is rotated around center
    center_shift = np.array((img.shape[0], img.shape[1])) / 2. - 0.5
    tform_center = SimilarityTransform(translation=-center_shift)
    tform_uncenter = SimilarityTransform(translation=center_shift)

    rotation = np.deg2rad(rotation)
    tform = AffineTransform(scale=(scale, scale), rotation=rotation, translation=(translate_y, translate_x))
    tform = tform_center + tform + tform_uncenter

    warped_img = warp(img, tform)
    return warped_img
