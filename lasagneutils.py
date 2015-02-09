"""Misc utility classes for nolearn / lasagne"""

import numpy as np

from nolearn.lasagne import BatchIterator as XBatchIterator

from skimage.transform import rotate, rescale
from skimage.morphology import disk
from skimage.filter import rank


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print('Early stopping.')
            print('Best valid loss was {:.6f} at epoch {}.'.format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


class StepDecay(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, net, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, net.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(net, self.name).set_value(new_value)


class BatchIterator(object):
    """https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne.py#L43
    Modified to ..."""
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) / bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb


class ShufflingBatchIteratorMixin(object):
    """From https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381"""
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShufflingBatchIteratorMixin, self).__iter__():
            yield res


class FlipBatchIterator(XBatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb


class FlippingBatchIteratorMixin(object):
    """Inspired by http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#data-augmentation
    Randomly flip 1/4 of the images
    """
    def transform(self, Xb, yb):
        Xb, yb = super(FlippingBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        # Select 1/4 images
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        # Xb = Xb[:, :, :, ::-1]

        return Xb, yb


class RotatingBatchIteratorMixin(object):
    """Randomly rotate 1/4 images"""
    def transform(self, Xb, yb):
        Xb, yb = super(RotatingBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        # angle_choices = range(0, 360, 45)
        # angle_choices = range(0, 360, 90)
        angle_choices = range(0, 360, 10)
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = np.array([[im_rotate(img, np.random.choice(angle_choices)) for img in c] for c in Xb[indices]])
        # Xb = np.array([[im_rotate(img, np.random.choice(angle_choices)) for img in c] for c in Xb])

        return Xb, yb


class ScalingBatchIteratorMixin(object):
    """Randomly scale for 1/4 images by 0.7 - 1.3"""
    def transform(self, Xb, yb):
        Xb, yb = super(ScalingBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        scale_choices = np.linspace(.9, 1.1, 20)
        Xb[indices] = np.array([[im_rescale(img, np.random.choice(scale_choices)) for img in c] for c in Xb[indices]])
        # Xb = np.array([[im_rescale(img, np.random.choice(scale_choices)) for img in c] for c in Xb])

        return Xb, yb


class TranslateBatchIteratorMixin(object):
    """Randomly move 1/4 images by -3 to 3 pixels"""
    def transform(self, Xb, yb):
        Xb, yb = super(TranslateBatchIteratorMixin, self).transform(Xb, yb)
        Xb, yb = np.copy(Xb), np.copy(yb)

        translate_choices = range(-5, 6)
        n = Xb.shape[0]
        indices = np.random.choice(n, n/2, replace=False)
        Xb[indices] = np.array([[im_translate(img, np.random.choice(translate_choices), np.random.choice(translate_choices)) for img in c] for c in Xb[indices]])
        # Xb = np.array([[im_translate(img, np.random.choice(translate_choices), np.random.choice(translate_choices)) for img in c] for c in Xb])

        return Xb, yb


# class MyBatchIterator(RotatingBatchIteratorMixin, ScalingBatchIteratorMixin, FlippingBatchIteratorMixin, BatchIterator):
class MyBatchIterator(RotatingBatchIteratorMixin, ScalingBatchIteratorMixin, TranslateBatchIteratorMixin, FlippingBatchIteratorMixin, ShufflingBatchIteratorMixin, BatchIterator):
    pass


def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


def float32(x):
    return np.cast['float32'](x)


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


def im_lcn(img, radius=30):
    """http://scikit-image.org/docs/dev/auto_examples/plot_local_equalize.html"""
    selem = disk(radius)
    img_eq = rank.equalize(img, selem=selem)
    return img_eq
