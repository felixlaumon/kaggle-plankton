from importlib import import_module


def get_net(fname):
    # TODO relative import is messy...
    return import_module('plankton.%s' % fname).net
