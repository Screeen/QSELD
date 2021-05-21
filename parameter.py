# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

import logging
logger = logging.getLogger(__name__)

def get_params(argv):
    params = dict(
        quick_test=False,    # To do quick test. Trains/test on small subset of dataset
        azi_only=False,      # Estimate Azimuth only

        # Dataset loading parameters
        dataset='ansim',    # Dataset to use: ansim, resim, cansim, cresim, real, mansim or mreal
        overlap=[1, 2],         # maximum number of overlapping sound events [1, 2, 3]
        train_split=[1, 2],     # Cross validation split [1, 2, 3]
        val_split=[3],
        test_split=[4],
        db=30,             # SNR of sound events.
        nfft=512,          # FFT/window length size
        debug_load_few_files=False,

        # DNN Model parameters
        sequence_length=512,        # Feature sequence length
        batch_size=16,               # Batch size (default 16)
        dropout_rate=0.0,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        pool_size=[8, 8, 2],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
        xyz_def_zero=True,          # Use default DOA Cartesian value x,y,z = 0,0,0
        nb_epochs=500,             # Train for maximum epochs

        epochs_per_iteration=2,
        doa_objective='mse',
        recurrent_type='tcn_new',  # TCN, GRU, tcn_new

        # TCN
        data_format='channels_last',
        spatial_dropout_rate=0.5,
        nb_tcn_filt=128,
        nb_tcn_blocks=10,
        use_quaternions=False,
        use_giusenso=False,

        # Not important
        mode='regr',        # Only regression ('regr') supported as of now
        nb_cnn3d_filt=0,   # For future. Not relevant for now
        cnn_3d=False,       # For future. Not relevant for now
        weakness=0          # For future. Not relevant for now
    )
    logger.info("SET: {}".format(argv))
    # ########### default parameters ##############
    params['patience'] = int(0.1 * params['nb_epochs'])     # Stop training if patience reached

    # ########### User defined parameters ##############
    params['use_quaternions'] = True if 'q' in argv else False
    argv = argv.replace('q', '')

    if argv == '1':
        logger.info("USING DEFAULT PARAMETERS\n")

    # Quick test
    elif '999' in argv:
        logger.info("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['nb_epochs'] = 2
        params['debug_load_few_files'] = False
        params['batch_size'] = 4


    elif argv == '888':
        logger.info("OVERFIT MODE\n")
        params['quick_test'] = True
        params['nb_epochs'] = 250
        params['batch_size'] = 4
        params['debug_load_few_files'] = True
        params['spatial_dropout_rate'] = 0
        params['dropout_rate'] = 0

    elif argv == '777':
        logger.info("Test evaluation\n")
        params['quick_test'] = True
        params['nb_epochs'] = 1
        params['debug_load_few_files'] = False

    # Different datasets
    elif argv == '2':  # anechoic simulated Ambisonic data set
        params['dataset'] = 'ansim'
        params['sequence_length'] = 256
        params['batch_size'] = 8
        # params['sequence_length'] = 512

    elif argv == '3':  # reverberant simulated Ambisonic data set
        params['dataset'] = 'resim'
        params['sequence_length'] = 256

    elif argv == '4':  # anechoic simulated circular-array data set
        params['dataset'] = 'cansim'
        params['sequence_length'] = 256

    elif argv == '5':  # reverberant simulated circular-array data set
        params['dataset'] = 'cresim'
        params['sequence_length'] = 256

    elif argv == '6':  # real-life Ambisonic data set
        params['dataset'] = 'real'
        params['sequence_length'] = 512

    # anechoic circular array data set split 1, overlap 3
    elif argv == '7':  #
        params['dataset'] = 'cansim'
        params['overlap'] = 3
        params['split'] = 1

    # anechoic Ambisonic data set with sequence length 64 and batch size 32
    elif argv == '8':  #
        params['dataset'] = 'ansim'
        params['sequence_length'] = 64
        params['batch_size'] = 32

    else:
        logger.info('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        logger.info("{}: {}".format(key, value))
    return params
