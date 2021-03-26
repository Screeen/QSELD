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
import keras.backend as K


from quaternion.qdense import *
from quaternion.qconv import *


class Identity:
    def __call__(self, x, *args, **kwargs):
        return x


def temporal_block(inp, num_filters_gru=0, dropout=0, recurrent_type='gru', data_in=(),
                   input_data_format='channels_last', spatial_dropout_rate=0.5, nb_tcn_filt_dilated_=128,
                   nb_tcn_blocks_=10, use_quaternions_=False):
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

    elif recurrent_type == 'TCN':

        nb_tcn_filters_dilated = nb_tcn_filt_dilated_
        nb_1x1_filters = 256
        nb_1x1_filters_final = 128  # FNN contents, number of nodes

        if use_quaternions_:
            nb_tcn_filters_dilated = nb_tcn_filters_dilated // 4
            nb_1x1_filters = nb_1x1_filters // 4
            nb_1x1_filters_final = nb_1x1_filters_final // 4
            ConvGeneric1D = QuaternionConv1D
            BatchNormGeneric = BatchNormalization
        else:
            ConvGeneric1D = Conv1D
            BatchNormGeneric = BatchNormalization

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
            print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
            inp = Reshape((num_frames, -1))(inp)
            print(f"K.int_shape(inp) {K.int_shape(inp)}")

        num_tcn_blocks = nb_tcn_blocks_

        d = [2 ** exp for exp in range(0, num_tcn_blocks)]  # list of dilation factors
        d = list(filter(lambda x: x <= num_frames, d))  # remove dilation factors larger than input

        skip_outputs = []
        layer_input = inp
        for idx, dil_rate in enumerate(d):
            spec_tcn_left = ConvGeneric1D(filters=nb_tcn_filters_dilated, kernel_size=(3), padding='same',
                                     dilation_rate=dil_rate,
                                     data_format=tcn_data_format)(layer_input)
            spec_tcn_left = BatchNormGeneric()(spec_tcn_left)

            # activations
            tanh_out = Activation('tanh')(spec_tcn_left)
            sigm_out = Activation('sigmoid')(spec_tcn_left)
            spec_act = keras.layers.Multiply()([tanh_out, sigm_out])

            # spatial dropout
            spec_act = keras.layers.SpatialDropout1D(rate=spatial_dropout_rate)(spec_act)

            # 1D convolution
            skip_out = ConvGeneric1D(filters=nb_1x1_filters, kernel_size=(1), padding='same',
                                     data_format=tcn_data_format)(spec_act)
            res_output = keras.layers.Add()([layer_input, skip_out])

            skip_outputs.append(skip_out)

            layer_input = res_output
        # ---------------------------------------

        # Residual blocks sum
        h = keras.layers.Add()(skip_outputs)
        h = Activation('relu')(h)

        print(f"K.int_shape(h) {K.int_shape(h)}")

        # 1D convolution
        h = ConvGeneric1D(filters=nb_1x1_filters_final, kernel_size=1, padding='same', data_format=tcn_data_format)(h)
        h = Activation('relu')(h)

        # 1D convolution
        h = ConvGeneric1D(filters=nb_1x1_filters_final, kernel_size=1, padding='same', data_format=tcn_data_format)(h)
        input_output = Activation('tanh')(h)

        print(f"K.int_shape(input_output) {K.int_shape(input_output)}")

        if input_data_format == 'channels_first':
            # Put temporal dimension first again
            input_output = Permute((2, 1))(input_output)
            print(f"K.int_shape(input_output) {K.int_shape(input_output)}")

        return input_output

    else:
        raise ValueError(f'recurrent_type must be gru or tcn, not {recurrent_type}')


def output_block(inp, out_shape, dropout=0, fnn_size=[0], dense_type='Dense'):
    # if dense_type == 'QDense':
    #     DenseGeneric = QuaternionDense
    # else:
    #     DenseGeneric = Dense

    DenseGeneric = Dense

    doa = inp
    for nb_fnn_filter in fnn_size:
        doa = TimeDistributed(DenseGeneric(nb_fnn_filter))(doa)
        doa = Dropout(dropout)(doa)

    num_units_doa = out_shape[1][-1]
    doa = TimeDistributed(Dense(num_units_doa))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # SED
    sed = inp
    for nb_fnn_filter in fnn_size:
        sed = TimeDistributed(DenseGeneric(nb_fnn_filter))(sed)
        sed = Dropout(dropout)(sed)
    num_units_sed = out_shape[0][-1]
    sed = TimeDistributed(Dense(num_units_sed))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    return sed, doa


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
              rnn_size, fnn_size, weights, params, data_format='channels_first'):
    # model definition
    keras.backend.set_image_data_format(data_format)
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    ##
    spatial_dropout = params['spatial_dropout_rate']
    recurrent_type = params['recurrent_type']
    nb_tcn_filt_ = params['nb_tcn_filt']  #num_conv_filters_tcn
    nb_tcn_blocks_ = params['nb_tcn_blocks']

    print(f"K.int_shape(spec_start) {K.int_shape(spec_start)}")
    spec_cnn = spec_start
    for i, this_pool_size in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, this_pool_size))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    spec_cnn = temporal_block(spec_cnn, num_filters_gru=rnn_size, dropout=dropout_rate,
                              recurrent_type=recurrent_type, data_in=data_in, input_data_format=data_format,
                              spatial_dropout_rate=spatial_dropout, nb_tcn_filt_dilated_=nb_tcn_filt_,
                              nb_tcn_blocks_=nb_tcn_blocks_)

    sed, doa = output_block(spec_cnn, out_shape=data_out, dropout=dropout_rate, fnn_size=fnn_size)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    # run_eagerly=True
    model.summary()
    return model


def get_model_quaternion(inp_shape, out_shape, params):
    inp = Input(shape=(inp_shape[-3], inp_shape[-2], inp_shape[-1]))
    print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")

    nb_cnn2d_filt = params['nb_cnn2d_filt'] // 4
    nb_tcn_filt_ = params['nb_tcn_filt']

    pool_size = params['pool_size']
    dropout_rate = params['dropout_rate']

    data_format = params['data_format']
    assert(data_format == "channels_last")

    fnn_size = params['fnn_size']
    loss_weights = params['loss_weights']
    spatial_do = params['spatial_dropout_rate']

    spec_cnn = inp
    for i, convCnt in enumerate(pool_size):
        if i == 0:
            spec_cnn = QuaternionConv2D(input_shape=inp_shape, filters=nb_cnn2d_filt, kernel_size=(3, 3),
                                        padding='same',
                                        data_format=data_format)(spec_cnn)
        else:
            spec_cnn = QuaternionConv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same',
                                        data_format=data_format)(spec_cnn)

        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(pool_size[i], 1))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    spec_cnn = temporal_block(spec_cnn, dropout=dropout_rate, input_data_format=data_format,
                              recurrent_type='TCN', data_in=inp_shape, spatial_dropout_rate=spatial_do,
                              nb_tcn_filt_dilated_=nb_tcn_filt_, nb_tcn_blocks_=params['nb_tcn_blocks'],
                              use_quaternions_=True)

    sed, doa = output_block(spec_cnn, out_shape=out_shape, dropout=dropout_rate, fnn_size=fnn_size,
                            dense_type='QDense')

    model = Model(inputs=inp, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=loss_weights)
    model.summary()
    return model
