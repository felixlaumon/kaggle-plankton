"""BatchNorm version 1: doesn't work"""

from time import time

import numpy as np
import theano
import theano.tensor as T

from lasagne import init
from lasagne import utils
from lasagne import nonlinearities
from lasagne.layers import Layer
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet as BaseNeuralNet


__all__ = [
    "BatchNormLayer",
    "NeuralNet"
]


class BatchNormLayer(Layer):
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, input_layer,
                 gamma=init.Uniform([0.95, 1.05]),
                 beta=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 epsilon=0.001,
                 **kwargs):
        super(BatchNormLayer, self).__init__(input_layer, **kwargs)
        self.additional_updates = None
        self.epsilon = epsilon
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        input_shape = input_layer.get_output_shape()

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

        self.mean_ema = theano.shared(
            np.zeros(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        self.variance_ema = theano.shared(
            np.ones(ema_shape, dtype=theano.config.floatX),
            borrow=True, broadcastable=ema_bc)

        self.batch_cnt = theano.shared(0)

    def get_params(self):
        return [self.gamma, self.beta]

    def get_state(self):
        return [self.gamma, self.beta,
                self.mean_ema, self.variance_ema, self.batch_cnt]

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, deterministic=False, *args, **kwargs):
        if deterministic:
            m = self.mean_ema
            v = self.variance_ema
        else:
            m = T.mean(input, self.axis, keepdims=True)
            v = T.sqrt(T.var(input, self.axis, keepdims=True) + self.epsilon)
            self.additional_updates = [
                # (self.mean_ema, self.mean_ema * 0.98 + m * 0.02),
                # (self.variance_ema, self.variance_ema * 0.98 + m * 0.02)]
                (self.mean_ema, self.mean_ema + m),
                (self.variance_ema, self.variance_ema + v),
                (self.batch_cnt, self.batch_cnt + 1)]

        input_norm = (input - m) / v
        y = self.gamma * input_norm + self.beta

        return self.nonlinearity(y)

    def get_additional_updates(self):
        if not self.additional_updates:
            raise RuntimeError
        return self.additional_updates

    def pre_train(self):
        self.mean_ema.set_value(np.zeros(
            self.mean_ema.get_value().shape,
            dtype=theano.config.floatX))
        self.variance_ema.set_value(np.zeros(
            self.variance_ema.get_value().shape,
            dtype=theano.config.floatX))
        self.batch_cnt.set_value(0)

    def post_train(self):
        new_mean = self.mean_ema.get_value() / self.batch_cnt.get_value()
        self.mean_ema.set_value(new_mean)
        new_var = self.variance_ema.get_value() / self.batch_cnt.get_value()
        self.variance_ema.set_value(new_var)


class NeuralNet(BaseNeuralNet):
    def _create_iter_funcs(self, output_layer, loss_func, update, input_type,
                           output_type):
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        loss_train = loss_func(
            output_layer.get_output(X_batch), y_batch)
        loss_eval = loss_func(
            output_layer.get_output(X_batch, deterministic=True), y_batch)
        predict_proba = output_layer.get_output(X_batch, deterministic=True)
        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        additional_updates = []
        for l in self.get_all_layers():
            if hasattr(l, 'get_additional_updates'):
                additional_updates += l.get_additional_updates()

        all_params = get_all_params(output_layer)
        update_params = self._get_params_for('update')
        updates = update(loss_train, all_params, **update_params)
        updates += additional_updates

        print(updates)

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
        """
        Modified version for train_loop for BatchNorm
        Taken from https://github.com/benanne/Lasagne/issues/141#issuecomment-75372139
        """
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        # MODIFIED
        # get layers that need pre_train
        pre_train_layers = []
        for l in self.get_all_layers():
            if hasattr(l, 'pre_train'):
                pre_train_layers.append(l)

        # MODIFIED
        # get layers that need post_train
        post_train_layers = []
        for l in self.get_all_layers():
            if hasattr(l, 'post_train'):
                post_train_layers.append(l)

        epoch = 0
        info = None
        best_valid_loss = np.inf
        best_train_loss = np.inf

        # MODIFIED
        for l in pre_train_layers:
            l.pre_train()

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

            # MODIFIED
            for l in post_train_layers:
                l.post_train()

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
