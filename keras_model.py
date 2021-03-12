#
# The SELDnet architecture
#
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, MaxPooling3D, Conv3D, merge, Conv1D
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.layers import GRU, GRUCell
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
# import keras
import keras.backend as K

from IPython import embed

from quaternion.qdense import *
from quaternion.qconv import *

class Identity:
    def __call__(self, x, *args, **kwargs):
        return x

def temporal_block(inp, num_filters_gru=0, dropout=0, recurrent_type='gru', data_in=(),
                   input_data_format='channels_first', spatial_dropout_rate=0.5):
    print(f'recurrent_type {recurrent_type}')
    print(f"temporal block input shape {K.int_shape(inp)}")

    if str.lower(recurrent_type) == 'gru':

        if input_data_format == 'channels_first':
            print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
            inp = Permute((2, 1, 3))(inp)
            print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
            inp = Reshape((data_in[-2], -1))(inp)
            print(f"K.int_shape(spec_rnn) {K.int_shape(inp)}")
        else:
            num_frames = data_in[1]
            inp = Reshape((num_frames, -1))(inp)
            print(f"K.int_shape(inp) {K.int_shape(inp)}")

        for idx, nb_rnn_filt in enumerate(num_filters_gru):
            inp = Bidirectional(
                GRU(nb_rnn_filt, activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                    return_sequences=True),
                merge_mode='mul'
            )(inp)
        return inp

    elif recurrent_type == 'TCN' or recurrent_type == 'QTCN':

        nb_tcn_filters_dilated = 128
        # nb_tcn_filters_dilated = 256
        nb_tcn_filters = 128
        nb_1x1_filters = 128  # FNN contents, number of nodes

        if recurrent_type == 'QTCN':
            nb_tcn_filters_dilated = nb_tcn_filters_dilated // 2
            nb_tcn_filters = nb_tcn_filters // 2
            nb_1x1_filters = nb_1x1_filters // 2
            ConvGeneric1D = QuaternionConv1D
            BatchNormGeneric = Identity
        else:
            ConvGeneric1D = Conv1D
            BatchNormGeneric = BatchNormalization

        num_tcn_blocks = 10

        d = [2 ** exp for exp in range(0, num_tcn_blocks)]  # list of dilation factors
        skip_outputs = []

        if input_data_format == 'channels_first':
            tcn_data_format = 'channels_first'
            print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
            inp = Permute((1, 3, 2))(inp)
            print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
            inp = Reshape((-1, data_in[-2]))(inp)
            print(f"K.int_shape(spec_rnn) {K.int_shape(inp)}")
        else:
            tcn_data_format = 'channels_last'
            num_frames = data_in[1]
            inp = Reshape((num_frames, -1))(inp)
            print(f"K.int_shape(inp) {K.int_shape(inp)}")

        layer_input = inp
        for idx, dil_rate in enumerate(d):
            spec_tcn = ConvGeneric1D(filters=nb_tcn_filters_dilated, kernel_size=3, padding='same', dilation_rate=dil_rate,
                              data_format=tcn_data_format)(layer_input)
            spec_tcn = BatchNormGeneric()(spec_tcn)
            tanh_out = Activation('tanh')(spec_tcn)
            sigm_out = Activation('sigmoid')(spec_tcn)
            spec_act = keras.layers.Multiply()([tanh_out, sigm_out])
            spec_act = keras.layers.SpatialDropout1D(rate=spatial_dropout_rate)(spec_act)
            skip_out = ConvGeneric1D(filters=nb_tcn_filters, kernel_size=1, padding='same',
                              data_format=tcn_data_format)(spec_act)
            # assert(K.int_shape(spec_resblock1) == K.int_shape(spec_drop1))
            # print(f"idx {idx}")
            # print(f"K.int_shape(layer_input) {K.int_shape(layer_input)}")
            # print(f"K.int_shape(spec_drop1) {K.int_shape(spec_drop1)}")
            # print(f"K.int_shape(skip_out) {K.int_shape(skip_out)}")
            skip_outputs.append(skip_out)
            res_output = keras.layers.Add()([layer_input, skip_out])
            layer_input = res_output

        print(f"K.int_shape(res_output) {K.int_shape(res_output)}")

        # Residual blocks sum
        h = keras.layers.Add()(skip_outputs)
        h = Activation('relu')(h)

        print(f"K.int_shape(h) {K.int_shape(h)}")

        # 1D convolution
        h = ConvGeneric1D(filters=nb_1x1_filters, kernel_size=1, padding='same', data_format=tcn_data_format)(h)
        h = Activation('relu')(h)

        # 1D convolution
        h = ConvGeneric1D(filters=nb_1x1_filters, kernel_size=1, padding='same', data_format=tcn_data_format)(h)
        input_output = Activation('tanh')(h)

        print(f"K.int_shape(input_output) {K.int_shape(input_output)}")

        if input_data_format == 'channels_first':
            # Put temporal dimension first again
            input_output = Permute((2, 1))(input_output)
            print(f"K.int_shape(input_output) {K.int_shape(input_output)}")

        return input_output


def output_block(inp, out_shape, dropout=0, fnn_size=[0], dense_type='Dense'):
    if dense_type == 'QDense':
        DenseGeneric = QuaternionDense
    else:
        DenseGeneric = Dense

    doa = inp
    for nb_fnn_filter in fnn_size:
        doa = TimeDistributed(DenseGeneric(nb_fnn_filter))(doa)
        doa = Dropout(dropout)(doa)

    doa = TimeDistributed(Dense(out_shape[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # SED
    sed = inp
    for nb_fnn_filter in fnn_size:
        sed = TimeDistributed(DenseGeneric(nb_fnn_filter))(sed)
        sed = Dropout(dropout)(sed)
    sed = TimeDistributed(Dense(out_shape[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    return sed, doa


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
              rnn_size, fnn_size, weights, params, data_format='channels_first'):

    # model definition
    keras.backend.set_image_data_format(data_format)
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    ##
    spatial_dropout = params['spatial_dropout']
    recurrent_type = params['recurrent_type']

    print(f"K.int_shape(spec_start) {K.int_shape(spec_start)}")
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    spec_cnn = temporal_block(spec_cnn, num_filters_gru=rnn_size, dropout=dropout_rate,
                              recurrent_type=recurrent_type, data_in=data_in, input_data_format=data_format,
                              spatial_dropout_rate=spatial_dropout)

    sed, doa = output_block(spec_cnn, out_shape=data_out, dropout=dropout_rate, fnn_size=fnn_size)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    model.summary()
    return model


from quaternion.qconv import *
def get_model_quaternion(inp_shape, out_shape, params):

    inp = Input(shape=(inp_shape[-3], inp_shape[-2], inp_shape[-1]))
    print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")

    nb_cnn2d_filt = params['nb_cnn2d_filt'] // 2

    pool_size = params['pool_size']
    dropout_rate = params['dropout_rate']
    data_format = params['data_format']
    fnn_size = params['fnn_size']
    loss_weights = params['loss_weights']
    spatial_do = params['spatial_dropout_rate']

    spec_cnn = inp
    for i, convCnt in enumerate(pool_size):
        if i == 0:
            spec_cnn = QuaternionConv2D(input_shape=inp_shape, filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same',
                                        data_format=data_format)(spec_cnn)
        else:
            spec_cnn = QuaternionConv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same',
                                        data_format=data_format)(spec_cnn)

        # spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    spec_cnn = temporal_block(spec_cnn, dropout=dropout_rate, input_data_format=data_format,
                              recurrent_type='QTCN', data_in=inp_shape, spatial_dropout_rate=spatial_do)

    sed, doa = output_block(spec_cnn, out_shape=out_shape, dropout=dropout_rate, fnn_size=fnn_size,
                            dense_type='QDense')

    model = Model(inputs=inp, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=loss_weights)
    model.summary()
    return model
