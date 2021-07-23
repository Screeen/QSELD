#
# The SELDnet architecture
#
import logging

from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Conv1D, Concatenate
from keras.layers import GRU
from keras.layers.core import Dense, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow import keras

from quaternion.qconv import *
from quaternion.qrecurrent import QuaternionGRU
from utils import make_list

logger = logging.getLogger(__name__)

global_num_classes = -1


class Identity:
    def __call__(self, x, *args, **kwargs):
        return x


def temporal_block_gru(inp, num_filters_gru=0, dropout=0, data_in=(), input_data_format='channels_last'):
    for idx, nb_rnn_filt in enumerate(make_list(num_filters_gru)):
        inp = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                return_sequences=True),
            merge_mode='mul'
        )(inp)
    return inp


def temporal_block_qgru(inp, num_filters_gru=0, dropout=0):
    for idx, nb_rnn_filt in enumerate(make_list(num_filters_gru)):
        inp = Bidirectional(
            QuaternionGRU(nb_rnn_filt, activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                          return_sequences=True),
            merge_mode='mul'
        )(inp)
    return inp


def temporal_block_new(inp, nb_tcn_filt_dilated_, nb_tcn_blocks_, spatial_dropout_rate, use_quaternions_,
                       data_in, input_data_format):

    assert (input_data_format == 'channels_last')
    tcn_data_format = 'channels_last'
    num_frames = data_in[1]
    logger.info(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
    inp = Reshape((num_frames, -1))(inp)
    logger.info(f"K.int_shape(inp) {K.int_shape(inp)}")

    num_tcn_blocks = nb_tcn_blocks_

    d = [2 ** exp for exp in range(0, num_tcn_blocks)]  # list of dilation factors
    d = list(filter(lambda x: x <= num_frames, d))  # remove dilation factors larger than input

    nb_tcn_filters_dilated = nb_tcn_filt_dilated_
    nb_1x1_filters = 128
    nb_1x1_filters_final = 128  # FNN contents, number of nodes

    if use_quaternions_:
        nb_tcn_filters_dilated = nb_tcn_filters_dilated // 2
        nb_1x1_filters = nb_1x1_filters // 4
        nb_1x1_filters_final = nb_1x1_filters_final // 2
        ConvGeneric1D = QuaternionConv1D
        BatchNormGeneric = BatchNormalization
    else:
        ConvGeneric1D = Conv1D
        BatchNormGeneric = BatchNormalization

    skip_outputs = []
    layer_input = inp
    for idx, dil_rate in enumerate(d):
        spec_tcn_left = ConvGeneric1D(filters=nb_tcn_filters_dilated, kernel_size=(3), padding='same',
                                      dilation_rate=dil_rate,
                                      data_format=tcn_data_format)(layer_input)
        spec_tcn_left = BatchNormGeneric()(spec_tcn_left)

        # activations
        spec_act = tf.keras.activations.relu(spec_tcn_left, alpha=0.2)

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
    h = tf.keras.activations.relu(h, alpha=0.2)

    logger.info(f"K.int_shape(h) {K.int_shape(h)}")

    # 1D convolution
    h = ConvGeneric1D(filters=nb_1x1_filters_final, kernel_size=1, padding='same', data_format=tcn_data_format)(h)
    h = tf.keras.activations.relu(h, alpha=0.2)

    # 1D convolution
    h = ConvGeneric1D(filters=nb_1x1_filters_final, kernel_size=1, padding='same', data_format=tcn_data_format)(h)
    input_output = tf.keras.activations.relu(h, alpha=0.2)
    # input_output = Activation('tanh')(h)

    logger.info(f"K.int_shape(input_output) {K.int_shape(input_output)}")

    if input_data_format == 'channels_first':
        # Put temporal dimension first again
        input_output = Permute((2, 1))(input_output)
        logger.info(f"K.int_shape(input_output) {K.int_shape(input_output)}")

    return input_output


def temporal_block(inp, num_filters_gru=0, dropout=0, recurrent_type='gru', data_in=(),
                   input_data_format='channels_last', spatial_dropout_rate=0.5, nb_tcn_filt_dilated_=128,
                   nb_tcn_blocks_=10, use_quaternions_=False):
    logger.info(f'recurrent_type {recurrent_type}')
    logger.info(f"temporal block input shape {K.int_shape(inp)}")

    if input_data_format == 'channels_first':
        logger.info(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
        inp = Permute((2, 1, 3))(inp)
        logger.info(f"K.int_shape(spec_cnn) {K.int_shape(inp)}")
        inp = Reshape((data_in[-2], -1))(inp)
        logger.info(f"K.int_shape(spec_rnn) {K.int_shape(inp)}")
    else:
        num_frames = data_in[1]
        inp = Reshape((num_frames, -1))(inp)
        logger.info(f"K.int_shape(inp) {K.int_shape(inp)}")

    recurrent_type = str.lower(recurrent_type)
    logger.info(f"Temporal block {recurrent_type} begins")
    if recurrent_type == 'gru':
        if use_quaternions_:
            input_output = temporal_block_qgru(inp, num_filters_gru, dropout)
        else:
            input_output = temporal_block_gru(inp, num_filters_gru, dropout, data_in, input_data_format)
    elif recurrent_type == 'tcn':
        input_output = temporal_block_guirguis(inp, nb_tcn_filt_dilated_, nb_tcn_blocks_, spatial_dropout_rate,
                                               use_quaternions_,
                                               data_in, input_data_format)
    elif recurrent_type == 'tcn_new':
        input_output = temporal_block_new(inp, nb_tcn_filt_dilated_, nb_tcn_blocks_, spatial_dropout_rate,
                                          use_quaternions_,
                                          data_in, input_data_format)
    else:
        raise ValueError(f'recurrent_type must be gru or tcn, not {recurrent_type}')

    return input_output


def output_block(inp, out_shape, dropout=0, fnn_size=[0], params=None):

    # SED
    sed = inp
    for nb_fnn_filter in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filter))(sed)
        sed = Dropout(dropout)(sed)
    num_units_sed = out_shape[0][-1]
    sed = TimeDistributed(Dense(num_units_sed))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    doa = inp
    for nb_fnn_filter in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filter))(doa)
        doa = Dropout(dropout)(doa)

    num_units_doa = num_units_sed*3
    # num_units_doa = out_shape[1][-1]
    doa = TimeDistributed(Dense(num_units_doa))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # for masked mse
    if params['doa_objective'] == 'masked_mse':
        doa = Concatenate()([keras.backend.zeros_like(sed), doa])

    return sed, doa


def get_model(input_shape, output_shape, dropout_rate, pool_size,
              rnn_size, fnn_size, weights, params, data_format='channels_first'):
    
    # model definition
    spec_start = Input(shape=(input_shape[-3], input_shape[-2], input_shape[-1]))

    ##
    spatial_dropout = params['spatial_dropout_rate']
    recurrent_type = params['recurrent_type']
    nb_tcn_filt_ = params['nb_tcn_filt']  # num_conv_filters_tcn
    nb_tcn_blocks_ = params['nb_tcn_blocks']
    use_quaternions_ = params['use_quaternions']

    assert (data_format == "channels_last")
    nb_cnn2d_filt = params['nb_cnn2d_filt'] if not use_quaternions_ else params['nb_cnn2d_filt'] // 4

    logger.info(f"K.int_shape(spec_start) {K.int_shape(spec_start)}")
    spec_cnn = spec_start
    for i, this_pool_size in enumerate(pool_size):
        if use_quaternions_:
            if i == 0:
                spec_cnn = QuaternionConv2D(input_shape=input_shape, filters=nb_cnn2d_filt, kernel_size=(3, 3),
                                            padding='same',
                                            data_format=data_format)(spec_cnn)
            else:
                spec_cnn = QuaternionConv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same',
                                            data_format=data_format)(spec_cnn)
        else:
            spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)

        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, this_pool_size))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)

    spec_cnn = temporal_block(spec_cnn, num_filters_gru=rnn_size, dropout=dropout_rate,
                              recurrent_type=recurrent_type, data_in=input_shape, input_data_format=data_format,
                              spatial_dropout_rate=spatial_dropout, nb_tcn_filt_dilated_=nb_tcn_filt_,
                              nb_tcn_blocks_=nb_tcn_blocks_, use_quaternions_=use_quaternions_)

    sed, doa = output_block(spec_cnn, out_shape=output_shape, dropout=dropout_rate, fnn_size=fnn_size, params=params)

    losses = ['binary_crossentropy', masked_mse] if params['doa_objective'] == 'masked_mse' \
        else ['binary_crossentropy', 'mse']
    model = Model(inputs=spec_start, outputs=[sed, doa])

    # disabling eager execution makes processing quicker
    eager_execution = True if params['quick_test'] else False
    model.compile(optimizer=Adam(), loss=losses, loss_weights=weights, run_eagerly=eager_execution)

    logger.info(model.summary())
    return model


"""
y_gt: shape (..., num_classes*3) for prediction with Cartesian coordinates.
sed_concat_doa_model_out: shape (..., num_classes + num_classes*3) for prediction with Cartesian coordinates.
"""
def masked_mse(sed_concat_doa_ground_truth, sed_concat_doa_model_out):
    # SED mask: Use the predicted DOAs only when gt SED > 0.5
    # logger.info(f"doa_ground_truth.shape {sed_concat_doa_ground_truth.shape}")
    # logger.info(f"sed_concat_doa_model_out.shape {sed_concat_doa_model_out.shape}")
    # logger.info(f"sed_concat_doa_ground_truth.shape {sed_concat_doa_ground_truth.shape}")
    
    sed_out_mask = sed_concat_doa_ground_truth[..., :global_num_classes] >= 0.5
    zeros_like_sed = keras.backend.zeros_like(sed_out_mask)
    sed_out_mask = keras.backend.repeat_elements(sed_out_mask, 3, -1)
    sed_out_mask = Concatenate()([zeros_like_sed, sed_out_mask])
    sed_out_mask = keras.backend.cast(sed_out_mask, 'float32')

    # Use the mask to computed mse. Normalize with the mask weights
    return keras.backend.sqrt(
        keras.backend.sum(
            keras.backend.square(sed_concat_doa_ground_truth - sed_concat_doa_model_out) * sed_out_mask)) \
           / keras.backend.sum(sed_out_mask)


def set_global_num_classes(params):
    global global_num_classes
    global_num_classes = params['num_classes']
    logger.info(f"Set num classes globally to {global_num_classes} to use it in custom loss function.")


def load_seld_model(model_file, doa_objective='mse', compileModel=True):
    logger.info(f"Model_file {model_file}, doa_objective {doa_objective}, compileModel {compileModel}")
    logger.info(f"global_num_classes {global_num_classes}")
    
    if doa_objective == 'mse':
        model = load_model(model_file, compile=compileModel)
    elif doa_objective == 'masked_mse':
        model = load_model(model_file, custom_objects={'masked_mse': masked_mse}, compile=compileModel)
    else:
        logger.error('ERROR: Unknown doa objective: {}'.format(doa_objective))
        exit()
    logger.info("Loaded successfuly")
    logger.info(model.summary())
    return model
