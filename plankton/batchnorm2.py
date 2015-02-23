"""Based on https://gist.github.com/skaae/8c7cca5e4b9621487738
and https://github.com/takacsg84/Lasagne/commit/d5545988e6484d1db4bb54bcfa541ba62e898829
Doesn't work either
"""

from time import time
import itertools

import numpy as np
import theano
import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities
from lasagne import utils
from lasagne.layers import Layer
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet as BaseNeuralNet

__all__ = [
    "BatchNormLayer"
]


class BatchNormLayer(Layer):
    """
    Batch normalization Layer [1]
    The user is required to setup updates for the learned parameters (Gamma
    and Beta). The values nessesary for creating the updates can be
    obtained by passing a dict as the moving_avg_hooks keyword to
    get_output().

    REF:
     [1] http://arxiv.org/abs/1502.03167

    :parameters:
        - input_layer : `Layer` instance
            The layer from which this layer will obtain its input

        - nonlinearity : callable or None (default: lasagne.nonlinearities.rectify)
            The nonlinearity that is applied to the layer activations. If None
            is provided, the layer will be linear.

        - epsilon : scalar float. Stabilizing training. Setting this too
            close to zero will result in nans.

    :usage:
        >>> from lasagne.layers import InputLayer, BatchNormLayer,
             DenseLayer
        >>> from lasagne.nonlinearities import linear, rectify
        >>> l_in = InputLayer((100, 20))
            l_dense = Denselayer(l_in, 50, nonlinearity=linear)
        >>> l_bn = BatchNormLayer(l_dense, nonlinearity=rectify)
        >>> hooks, input, updates = {}, T.matrix, []
        >>> l_out = l_bn.get_output(
              input, deterministic=False, moving_avg_hooks=hooks)
        >>> mulfac = 1.0/100.0
        >>> batchnormparams = list(itertools.chain(
              *[i[1] for i in hooks['BatchNormLayer:movingavg']]))
        >>> batchnormvalues = list(itertools.chain(
              *[i[0] for i in hooks['BatchNormLayer:movingavg']]))
        >>> for tensor, param in zip(tensors, params):
                updates.append((param, (1.0-mulfac)*param + mulfac*tensor))
            # append updates to your normal update list
    """
    def __init__(self, incoming,
                 gamma=init.Uniform([0.95, 1.05]),
                 beta=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 epsilon=0.001,
                 **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        input_shape = incoming.get_output_shape()
        self.epsilon = epsilon

        if len(input_shape) == 2:       # in case of dense layer
            self.axis = (0)
            param_shape = (input_shape[-1])
            self.gamma = self.create_param(gamma, param_shape)
            self.beta = self.create_param(beta, param_shape)
            ema_shape = (1, input_shape[-1])
            ema_bc = (True, False)
        elif len(input_shape) == 4:     # in case of conv2d layer
            self.axis = (0, 2, 3)
            param_shape = (input_shape[1], 1, 1)

            # it has to be made broadcastable on the first axis
            self.gamma = theano.shared(utils.floatX(gamma(param_shape)),
                                       broadcastable=(False, True, True),
                                       borrow=True)
            self.beta = theano.shared(utils.floatX(beta(param_shape)),
                                      broadcastable=(False, True, True),
                                      borrow=True)
            ema_shape = (1, input_shape[1], 1, 1)
            ema_bc = (True, False, True, True)
        else:
            raise NotImplementedError

        self.mean_inference = theano.shared(
            np.zeros(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)
        self.mean_inference.name = "shared:mean"

        self.variance_inference = theano.shared(
            np.ones(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)
        self.variance_inference.name = "shared:variance"

    def get_params(self):
        return [self.gamma, self.beta]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, moving_avg_hooks=None,
                       deterministic=False, *args, **kwargs):
        if deterministic is False:
            m = T.mean(input, axis=self.axis, keepdims=True)
            v = T.sqrt(T.var(input, axis=self.axis, keepdims=True) + self.epsilon)
            m.name = "tensor:mean"
            v.name = "tensor:variance"

            key = "BatchNormLayer:movingavg"
            if key not in moving_avg_hooks:
                moving_avg_hooks[key] = []
            moving_avg_hooks[key].append(
                [[m, v], [self.mean_inference, self.variance_inference]])
        else:
            m = self.mean_inference
            v = self.variance_inference

        input_norm = (input - m) / v                # normalize
        y = self.gamma * input_norm + self.beta     # scale and shift
        return self.nonlinearity(y)


class NeuralNet(BaseNeuralNet):
    def _create_iter_funcs(self, output_layer, loss_func, update, input_type,
                           output_type):
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        hooks = {}
        loss_train = loss_func(
            output_layer.get_output(X_batch, moving_avg_hooks=hooks), y_batch)
        loss_eval = loss_func(
            output_layer.get_output(X_batch, deterministic=True), y_batch)
        predict_proba = output_layer.get_output(X_batch, deterministic=True)

        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = get_all_params(output_layer)
        update_params = self._get_params_for('update')
        batchnormparams = list(itertools.chain(*[i[1] for i in hooks['BatchNormLayer:movingavg']]))
        batchnormvalues = list(itertools.chain(*[i[0] for i in hooks['BatchNormLayer:movingavg']]))
        batchupd = get_batchnorm_updates(batchnormvalues, batchnormparams, 100)
        updates = update(loss_train, all_params, **update_params)
        updates += batchupd

        train_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_train],
            updates=updates,
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        eval_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_eval, accuracy],
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        predict_iter = theano.function(
            inputs=[theano.Param(X_batch)],
            outputs=predict_proba,
            givens={
                X: X_batch,
                },
            )

        return train_iter, eval_iter, predict_iter

    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        info = None
        best_valid_loss = np.inf
        best_train_loss = np.inf

        if self.verbose:
            print("""
 Epoch  |  Train loss  |  Valid loss  |  Train / Val  |  Valid acc  |  Dur
--------|--------------|--------------|---------------|-------------|-------\
""")

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            if self.verbose:
                best_train = best_train_loss == avg_train_loss
                best_valid = best_valid_loss == avg_valid_loss
                print(" {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  "
                      "|  {:>11.6f}  |  {:>9}  |  {:>3.1f}s".format(
                          epoch,
                          ansi.BLUE if best_train else "",
                          avg_train_loss,
                          ansi.ENDC if best_train else "",
                          ansi.GREEN if best_valid else "",
                          avg_valid_loss,
                          ansi.ENDC if best_valid else "",
                          avg_train_loss / avg_valid_loss,
                          "{:.2f}%".format(avg_valid_accuracy * 100)
                          if not self.regression else "",
                          time() - t0,
                          ))

            info = dict(
                epoch=epoch,
                train_loss=avg_train_loss,
                valid_loss=avg_valid_loss,
                valid_accuracy=avg_valid_accuracy,
                )
            self.train_history_.append(info)
            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)


class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


def get_batchnorm_updates(tensors, params, avglen):
        updates = []
        mulfac = 1.0 / avglen
        for tensor, param in zip(tensors, params):
            updates.append((param, (1.0 - mulfac)*param + mulfac * tensor))
        return updates
