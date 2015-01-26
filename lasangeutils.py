"""Misc utility classes for nolearn / lasange"""

import numpy as np

from nolearn.lasagne import BatchIterator

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


class AdjustVariable(object):
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


class ShufflingBatchIteratorMixin(object):
    """From https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381"""
    def __iter__(self):
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShufflingBatchIteratorMixin, self).__iter__():
            yield res


class FlippingBatchIteratorMixin(object):
    """Inspired by http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/#data-augmentation
    Randomly flip 1/5 of the images
    """
    def transform(self, Xb, yb):
        Xb, yb = super(FlippingBatchIteratorMixin, self).transform(Xb, yb)

        # Select 1/5 images
        n = Xb.shape[0]
        indices = np.random.choice(n, n/5, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb


class RotatingBatchIteratorMixin(object):
    """Randomly rotate all images"""
    def transform(self, Xb, yb):
        Xb, yb = super(FlippingBatchIteratorMixin, self).transform(Xb, yb)

        # TODO Select 1/5 images only?
        Xb = np.array([random_rotate_img(img) for img in Xb[indices]])

        return Xb, yb


class MyBatchIterator(BatchIterator, ShufflingBatchIteratorMixin, FlippingBatchIteratorMixin, RotatingBatchIteratorMixin):
    pass


def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


def float32(x):
    return np.cast['float32'](x)


def random_rotate_img(img, angle_range=360):
    """Randomly rotate the image"""
    angle = np.random.randint(angle_range)
    # Fill "blank space" with white because the images have white background
    # instead
    return rotate(img, angle, cval=1.0)


def im_rotate(img, angle):
    return skimage.transform.rotate(img, angle, mode='reflect')


def im_translate(img, shift_x, shift_y):
    ## this could probably be a lot easier... meh.
    # downsampling afterwards is recommended
    translate_img = np.ones_like(img, dtype=img.dtype)

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

    return translate_img


def im_rescale(img, scale_factor):
    zoomed_img = np.ones_like(img, dtype=img.dtype)
    zoomed = skimage.transform.rescale(img, scale_factor)

    if scale_factor >= 1.0:
        shift_x = (zoomed.shape[0] - img.shape[0]) // 2
        shift_y = (zoomed.shape[1] - img.shape[1]) // 2
        zoomed_img[:,:] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
    else:
        shift_x = (img.shape[0] - zoomed.shape[0]) // 2
        shift_y = (img.shape[1] - zoomed.shape[1]) // 2
        zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

    return zoomed_img


def im_lcn(img, radius=30):
    """http://scikit-image.org/docs/dev/auto_examples/plot_local_equalize.html"""
    selem = skimage.morphology.disk(radius)
    img_eq = skimage.filter.rank.equalize(img, selem=selem)
    return img_eq
