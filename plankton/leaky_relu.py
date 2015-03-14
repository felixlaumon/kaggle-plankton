"""
Taken directly from
https://github.com/benanne/Lasagne/pull/163
"""

from lasagne.nonlinearities import rectify

# leaky rectify
# Maas et al: Rectifier Nonlinearities Improve Neural Network Acoustic Models
# http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
class LeakyRectify(object):

    def __init__(self, leakiness=0.01):
        self.leakiness = leakiness

    def __call__(self, x):
        if self.leakiness:
            import theano.tensor as T
            return T.maximum(self.leakiness * x, x)
        else:
            return rectify(x)

leaky_rectify = LeakyRectify()  # shortcut with default leakiness

