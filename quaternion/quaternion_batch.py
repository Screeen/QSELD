from keras.layers import Layer
from tensorflow.python.framework import tensor_shape
import tensorflow as tf
from tensorflow.python.keras.engine.input_spec import InputSpec


class QuaternionBatchNorm(Layer):

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.eps = tf.constant(1e-5)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        axis_to_dim = {x: input_shape.dims[x].value for x in [self.axis]}
        for x in axis_to_dim:
          if axis_to_dim[x] is None:
            raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                             input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        # Single axis batch norm (most common/default use-case)
        param_shape = (list(axis_to_dim.values())[0],)
        gamma_shape = ([p//4 for p in param_shape])
        # print(f"param_shape:{param_shape}")
        # print(f"gamma_shape:{gamma_shape}")

        self.gamma = self.add_weight(
            name='gamma',
            shape=gamma_shape,
            # dtype=self._param_dtype,
            initializer='ones',
            trainable=True)

        self.beta = self.add_weight(
            name='beta',
            shape=param_shape,
            # dtype=self._param_dtype,
            initializer='zeros',
            trainable=True)

        # Create moving_mean and moving_variance
        # self.moving_mean = self.add_weight(name='moving_mean', shape=param_shape, trainable=False, initializer='zeros')
        # self.moving_variance = self.add_weight(name='moving_variance', shape=gamma_shape, trainable=False, initializer='ones')


    def call(self, inputs, is_training=True, **kwargs):
        # quat_components = torch.chunk(input, 4, dim=1)

        quat_components = tf.split(value=inputs, num_or_size_splits=4, axis=self.axis)

        quat_mean = [tf.reduce_mean(x) for x in quat_components]
        # print(f"quat_mean: {[tf.print(x) for x in quat_mean]}")

        deltas = [x - mu for (x, mu) in zip(quat_components, quat_mean)]
        # print(f"deltas: {deltas}")
        quat_variance = tf.reduce_mean(tf.reduce_sum([delta ** 2 for delta in deltas]))

        deltas = [x - mu for (x, mu) in zip(quat_components, quat_mean)]

        # print(f"quat_variance: {tf.print(quat_variance)}")
        denominator = tf.sqrt(quat_variance + self.eps)

        normalized_components = [delta/denominator for delta in deltas]
        # print(f"normalized_components: {normalized_components}")

        beta_components = tf.split(value=self.beta, num_or_size_splits=4, axis=self.axis)
        # print(f"self.beta: {self.beta}")
        # print(f"beta_components: {beta_components}")
        # print(f"self.gamma: {self.gamma}")
        # Multiply gamma (stretch scale) and add beta (shift scale)
        out = [(self.gamma * normalized) + beta for (normalized, beta) in zip(normalized_components, beta_components)]
        # print(f"out: {out}")
        new_out = tf.concat(values=out, axis=self.axis)
        # print(f"new_out: {new_out}")
        return new_out
