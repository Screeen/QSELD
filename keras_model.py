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

keras.backend.set_image_data_format('channels_first')
from IPython import embed


def temporal_block(inp, num_filters_temporal, dropout=0, recurrent_type='gru', data_in=()):
    print(f"temporal block input shape {K.int_shape(inp)}")

    if recurrent_type == 'gru':

        print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
        inp = Permute((2, 1, 3))(inp)
        print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
        inp = Reshape((data_in[-2], -1))(inp)
        print(f"K.int_shape(spec_rnn) {K.int_shape(inp)}")

        for idx, nb_rnn_filt in enumerate(num_filters_temporal):
            inp = Bidirectional(
                GRU(nb_rnn_filt, activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                    return_sequences=True),
                merge_mode='mul'
            )(inp)
        return inp

    elif recurrent_type == 'TCN':
        num_tcn_blocks = 10
        nb_tcn_filters_dilated = 256
        nb_tcn_filters = 128
        nb_1x1_filters = 128  # FNN contents, number of nodes
        data_format = 'channels_first'
        d = [2 ** exp for exp in range(0, num_tcn_blocks)]  # list of dilation factors
        skip_outputs = []

        #####
        spatial_dropout_rate = 0
        ####

        print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
        inp = Permute((1, 3, 2))(inp)
        print(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
        inp = Reshape((-1, data_in[-2]))(inp)
        print(f"K.int_shape(spec_rnn) {K.int_shape(inp)}")

        layer_input = inp
        for idx, dil_rate in enumerate(d):
            spec_tcn = Conv1D(filters=nb_tcn_filters_dilated, kernel_size=3, padding='same', dilation_rate=dil_rate,
                              data_format=data_format)(layer_input)
            spec_tcn = BatchNormalization()(spec_tcn)
            tanh_out = Activation('tanh')(spec_tcn)
            sigm_out = Activation('sigmoid')(spec_tcn)
            spec_act = keras.layers.Multiply()([tanh_out, sigm_out])
            if spatial_dropout_rate > 0:
                spec_act = keras.layers.SpatialDropout1D(rate=spatial_dropout_rate)(spec_act)
            skip_out = Conv1D(filters=nb_tcn_filters, kernel_size=1, padding='same',
                              data_format=data_format)(spec_act)
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
        skip_outputs = []
        h = Activation('relu')(h)

        print(f"K.int_shape(h) {K.int_shape(h)}")

        # 1D convolution
        h = Conv1D(filters=nb_1x1_filters, kernel_size=1, padding='same', data_format=data_format)(h)
        h = Activation('relu')(h)

        # 1D convolution
        h = Conv1D(filters=nb_1x1_filters, kernel_size=1, padding='same', data_format=data_format)(h)
        input_output = Activation('tanh')(h)

        print(f"K.int_shape(input_output) {K.int_shape(input_output)}")

        # Put temporal dimension first again
        input_output = Permute((2, 1))(input_output)
        print(f"K.int_shape(input_output) {K.int_shape(input_output)}")

        return input_output


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
              rnn_size, fnn_size, classification_mode, weights, recurrent_type='gru'):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    print(f"K.int_shape(spec_start) {K.int_shape(spec_start)}")
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    spec_cnn = temporal_block(spec_cnn, num_filters_temporal=rnn_size, dropout=dropout_rate,
                              recurrent_type='TCN', data_in=data_in)

    doa = spec_cnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # SED
    sed = spec_cnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    return model
