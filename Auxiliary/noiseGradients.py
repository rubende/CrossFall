# From https://github.com/cpury/keras_gradient_noise/
import inspect
import keras
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K


def _get_shape(x):
    if hasattr(x, 'dense_shape'):
        return x.dense_shape

    return K.shape(x)


def add_gradient_noise(BaseOptimizer):
    """
    Given a Keras-compatible optimizer class, returns a modified class that
    supports adding gradient noise as introduced in this paper:
    https://arxiv.org/abs/1511.06807
    The relevant parameters from equation 1 in the paper can be set via
    noise_eta and noise_gamma, set by default to 0.3 and 0.55 respectively.
    """
    if not (
            inspect.isclass(BaseOptimizer) and
            issubclass(BaseOptimizer, keras.optimizers.Optimizer)
    ):
        raise ValueError(
            'add_gradient_noise() expects a valid Keras optimizer'
        )

    class NoisyOptimizer(BaseOptimizer):
        def __init__(self, noise_eta=0.3, noise_gamma=0.55, weight_decay = 0.0001, bpe = 1, **kwargs):
            super(NoisyOptimizer, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.noise_eta = K.variable(noise_eta, name='noise_eta')
                self.noise_gamma = K.variable(noise_gamma, name='noise_gamma')
                self.weight_decay = K.variable(weight_decay, name='weight_decay')
                self.bpe = K.variable(bpe, name='bpe')


        def get_gradients(self, loss, params):
            grads = super(NoisyOptimizer, self).get_gradients(loss, params)

            # Add decayed gaussian noise
            t = tf.math.floor(K.cast(self.iterations, 'float32') / self.bpe)
            t = K.cast(t, K.dtype(grads[0]))
            variance = self.noise_eta / ((1 + t) ** self.noise_gamma)
            for i in range(len(grads)):
                if grads[i].op.inputs[0].name.find('bn') < 0:
                    penalty = math_ops.reduce_sum(self.weight_decay * math_ops.square(params[i]))
                    grads[i] = grads[i] + (variance * K.random_normal(_get_shape(grads[i]), mean=0.0, stddev=1.0, dtype=K.dtype(grads[i]))) + penalty * params[i]



            return grads

        def get_config(self):
            config = {'noise_eta': float(K.get_value(self.noise_eta)),
                       'noise_gamma': float(K.get_value(self.noise_gamma)),'weight_decay':float(K.get_value(self.weight_decay)), 'bpe':int(K.get_value(self.bpe))}
            base_config = super(NoisyOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    NoisyOptimizer.__name__ = 'Noisy{}'.format(BaseOptimizer.__name__)

    return NoisyOptimizer
