# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras.layers
import keras.layers.convolutional
from math import prod
from quaternion.qconv import *


class DummyLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return inputs

def hamilton_product(q0, q1):
    """
    Applies a Hamilton product q0 * q1:
    Shape:
        - q0, q1 should be (..., quaternion_number)
        (rr' - xx' - yy' - zz')  +
        (rx' + xr' + yz' - zy')i +
        (ry' - xz' + yr' + zx')j +
        (rz' + xy' - yx' + zr')k +
    """
    qs = [q0, q1]
    for idx, x in enumerate(qs):
        qs[idx] = np.reshape(qs[idx], newshape=[-1, 4])
        qs[idx] = np.squeeze(qs[idx])
    q0, q1 = qs

    r0, x0, y0, z0 = tuple(q0[..., idx] for idx in range(4))
    r1, x1, y1, z1 = tuple(q1[..., idx] for idx in range(4))

    q0_matrix = np.array([[r0, -x0, -y0, -z0], [x0, r0, -z0, y0], [y0, z0, r0, -x0], [z0, -y0, x0, r0]])
    q1_matrix = np.array([[r1, -x1, -y1, -z1], [x1, r1, -z1, y1], [y1, z1, r1, -x1], [z1, -y1, x1, r1]])
    q_out = np.matmul(q0_matrix, q1_matrix)

    print(f"q0: {q0}, q1: {q1}")
    print(f"q_out: {q_out[..., 0]}")

    return np.reshape(q_out[..., 0], newshape=q0.shape)

def q_mult(q1, q2):

    qs = [q1, q2]
    for idx, x in enumerate(qs):
        qs[idx] = np.reshape(qs[idx], newshape=[-1, 4])
        qs[idx] = np.squeeze(qs[idx])
    q1, q2 = qs

    print(f"q1:{q1}")
    print(f"q2:{q2}")

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    print(f"qout_doublecheck: {np.array([w, x, y, z])}")

    return w, x, y, z

def main():

    np.set_printoptions(precision=2)
    np.random.seed(0)

    input_generators = [np.arange,np.arange,np.arange,np.arange,]
    my_input_shapes = [(1, 4), (1, 8),(1, 4), (1, 8),]
    output_filters = [4]
    num_layers = [1, 1, 1, 1]

    for (my_input_shape, input_gen, output_filter, num_layer) in \
            zip(my_input_shapes, input_generators, output_filters, num_layers):

        GenericConv1D = keras.layers.convolutional.Conv1D

        # GenericConv1D = QuaternionConv1D
        # output_filter = output_filter // 4

        x = keras.layers.Input(shape=my_input_shape)
        q1 = GenericConv1D(filters=output_filter, kernel_size=1, use_bias=False,
                               padding='valid', bias_initializer='zero', kernel_initializer='ones',
                           data_format='channels_last')
        h = q1(x)

        for idx in range(num_layer-1):
            h = GenericConv1D(filters=1, kernel_size=1, use_bias=False, data_format='channels_last',
                                  padding='valid', bias_initializer='zero', kernel_initializer='ones')(h)

        out = h
        mlp_model = keras.models.Model(inputs=[x], outputs=[out], name="QuaternionConvTest")
        mlp_model.compile('adam', 'mse')

        if input_gen == np.arange:
            my_input = input_gen(1, prod(my_input_shape) + 1).reshape((1,) + my_input_shape)
        else:
            my_input = input_gen((1,) + my_input_shape)

        prediction = mlp_model.predict(my_input, verbose=0)

        print("\n")
        print(f'## Quaternion 1x1 conv ({num_layer} layers), {output_filter} output filters:')
        print(mlp_model.summary())
        print(f"my_input:{my_input}")
        print(f"kernel:{q1.get_weights()[0]}")
        print(f"DNN prediction: {prediction}")

        ###
        print('## Hamilton product (matrix):')
        for my_input_mod in np.split(my_input, indices_or_sections=my_input.shape[-1]//4, axis=-1):
            fake_kernel = np.ones_like(my_input_mod)
            qout = hamilton_product(my_input_mod, fake_kernel)
            for idx in range(num_layer-1):
                qout = hamilton_product(qout, fake_kernel)

import sys

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, IOError) as e:
        sys.exit(e)
