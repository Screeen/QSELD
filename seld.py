#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

# Use GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_data_generator
import evaluation_metrics
import network
import parameter
import utils
import keras.utils
import time
import datetime
import keras_model_giusenso
import keras.backend as K

import logging
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from utils import list_to_string

plot.switch_backend('agg')

np.set_printoptions(precision=1, suppress=True, floatmode='fixed')


def collect_test_labels(data_generator, data_shape, classification_mode, quick_test):
    logger.info(f"Collecting ground truth for test data")
    nb_batch = data_generator.get_total_batches_in_data()

    batch_size = data_shape[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, data_shape[0][1], data_shape[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, data_shape[0][1], data_shape[1][2]))

    logger.info(f"gt_sed.shape: {gt_sed.shape}")
    logger.info(f"gt_doa.shape: {gt_doa.shape}")
    logger.info("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for _, tmp_label in data_generator.generate():
        if(cnt % 25 == 0):
            logger.info(f"Batch {cnt}")
        gt_sed_batch = tmp_label[0] # 16, 512, 11 - batch size, fftsize, ...
        gt_doa_batch = tmp_label[1] # 16, 512, 33 - batch size, fftsize, ...
        gt_sed[cnt * batch_size : (cnt + 1) * batch_size, ...] = gt_sed_batch
        gt_doa[cnt * batch_size : (cnt + 1) * batch_size, ...] = gt_doa_batch
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa


def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _epoch_metric_loss):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(311)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(nb_epoch), _epoch_metric_loss, label='metric')
    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def predict_single_batch(model, data_gen_train):

    batch = next(data_gen_train.generate())
    pred = model.predict(
        x=batch[0],
        steps=1,
        verbose=2
    )

    for batch_idx in (0, 1, 2):
        for temporal_idx in (100, 101, 102):
            gt_sed = batch[1][0][batch_idx][temporal_idx]
            gt_doa = batch[1][1][batch_idx][temporal_idx]
            pred_sed = pred[0][batch_idx][temporal_idx]
            pred_doa = pred[1][batch_idx][temporal_idx]

            logger.debug(f"pred_sed {pred_sed}")
            logger.debug(f"gt_sed   {gt_sed}")
            logger.debug(f"pred_doa\n {pred_doa}")
            logger.debug(f"gt_doa  \n {gt_doa}")


def collect_ground_truth(data_gen, params):
    data_in, data_out = data_gen.get_data_sizes()
    gt = collect_test_labels(data_gen, data_out, params['mode'], params['quick_test'])
    sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
    doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

    return sed_gt, doa_gt


def train(model, data_gen_train, data_gen_val, params, log_dir=".", unique_name="unique_name"):
    logger.info("Train function called")
    sed_gt, doa_gt = collect_ground_truth(data_gen_val, params)

    best_metric = 99999
    conf_mat = None
    best_conf_mat = None
    best_epoch = -1
    patience_cnt = 0
    epoch_metric_loss = np.zeros(params['nb_epochs'])
    tr_loss = np.zeros(params['nb_epochs'])
    val_loss = np.zeros(params['nb_epochs'])
    doa_loss = np.zeros((params['nb_epochs'], 6))
    sed_loss = np.zeros((params['nb_epochs'], 2))
    nb_epoch = params['nb_epochs']
    model_path = os.path.join(log_dir, 'model')

    K.clear_session()

    for epoch_cnt in range(nb_epoch):
        logger.info(f"Iteration {epoch_cnt}/{nb_epoch}")
        start = time.time()
        hist = model.fit(
            x=data_gen_train.generate(),
            steps_per_epoch=data_gen_train.get_total_batches_in_data(),
            validation_data=data_gen_val.generate(),
            validation_steps=data_gen_val.get_total_batches_in_data(),
            epochs=params['epochs_per_iteration'],
            verbose=2,
            # callbacks=[MyCustomCallback]
        )

        tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
        val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

        if (params['debug_load_single_batch']) and (epoch_cnt % 10 != 0) and (epoch_cnt != nb_epoch - 1):
            plot_functions(os.path.join(log_dir, 'training_curves'), tr_loss, val_loss, sed_loss, doa_loss,
                           epoch_metric_loss)
        else:
            predict_single_batch(model, data_gen_train)

            pred = model.predict(
                x=data_gen_val.generate(),
                steps=data_gen_val.get_total_batches_in_data(),
                verbose=2
            )
            if params['mode'] == 'regr':
                sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5

                doa_pred = pred[1]
                num_classes = sed_pred.shape[-1]
                num_dims_xyz = 3

                if doa_pred.shape[-1] > num_classes * num_dims_xyz:  # true means we are using masked mse
                    doa_pred = doa_pred[..., num_classes:]
                    logger.debug(f"doa_pred.shape {doa_pred.shape}")

                if doa_gt.shape[-1] > num_classes * num_dims_xyz:  # true means we are using masked mse
                    doa_gt = doa_gt[..., num_classes:]
                    logger.debug(f"doa_gt.shape {doa_gt.shape}")

                doa_pred = evaluation_metrics.reshape_3Dto2D(doa_pred)

                sed_loss[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt,
                                                                               data_gen_val.nb_frames_1s())
                if params['azi_only']:
                    doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred,
                                                                                                     doa_gt,
                                                                                                     sed_pred,
                                                                                                     sed_gt)
                else:
                    doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred,
                                                                                                      doa_gt,
                                                                                                      sed_pred,
                                                                                                      sed_gt)

                epoch_metric_loss[epoch_cnt] = np.mean([
                    sed_loss[epoch_cnt, 0],
                    1 - sed_loss[epoch_cnt, 1],
                    2 * np.arcsin(doa_loss[epoch_cnt, 1] / 2.0) / np.pi,
                    1 - (doa_loss[epoch_cnt, 5] / float(doa_gt.shape[0]))]
                )

            plot_functions(os.path.join(log_dir, 'training_curves'), tr_loss, val_loss, sed_loss, doa_loss,
                           epoch_metric_loss)

            patience_cnt += 1
            if (epoch_metric_loss[epoch_cnt] < best_metric):
                best_metric = epoch_metric_loss[epoch_cnt]
                best_conf_mat = conf_mat
                best_epoch = epoch_cnt
                model.save(model_path)
                patience_cnt = 0

            logger.info(
                'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
                'F1_overall: %.2f, ER_overall: %.2f, '
                'doa_error_gt: %.2f, doa_error_pred: %.2f, good_pks_ratio:%.2f, '
                'error_metric: %.2f, best_error_metric: %.2f, best_epoch : %d' %
                (
                    epoch_cnt, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    sed_loss[epoch_cnt, 1], sed_loss[epoch_cnt, 0],
                    doa_loss[epoch_cnt, 1], doa_loss[epoch_cnt, 2], doa_loss[epoch_cnt, 5] / float(sed_gt.shape[0]),
                    epoch_metric_loss[epoch_cnt], best_metric, best_epoch
                )
            )

            if patience_cnt > params['patience']:
                break

        # otherwise RAM use increases after every epoch. But is the optimizer state forgotten?
        K.clear_session()

        if params['debug_load_single_batch'] and hist.history.get('loss')[-1] < 0.01:
            break

    if params['debug_load_single_batch']:
        model.save(model_path)

    else:
        logger.info('best_conf_mat : {}'.format(best_conf_mat))
        logger.info('best_conf_mat_diag : {}'.format(np.diag(best_conf_mat)))
        logger.info('saved model for the best_epoch: {} with best_metric: {},  '.format(best_epoch, best_metric))
        logger.info('DOA Metrics: doa_loss_gt: {}, doa_loss_pred: {}, good_pks_ratio: {}'.format(
            doa_loss[best_epoch, 1], doa_loss[best_epoch, 2], doa_loss[best_epoch, 5] / float(sed_gt.shape[0])))
        logger.info(
            'SED Metrics: F1_overall: {}, ER_overall: {}'.format(sed_loss[best_epoch, 1], sed_loss[best_epoch, 0]))

    np.save(os.path.join(log_dir, 'training-loss'), [tr_loss, val_loss])
    logger.info(f'unique_name: {unique_name}')
    logger.info(f'log_dir: {log_dir}')
    # predict_single_batch(model, data_gen_train)


def evaluate(model, data_gen_test, params, log_dir=".", unique_name="unique_name"):
    logger.info("EVALUATE function called")
    sed_gt, doa_gt = collect_ground_truth(data_gen_test, params)

    predict_single_batch(model, data_gen_test)

    dnn_output = model.predict(
        x=data_gen_test.generate(),
        steps=data_gen_test.get_total_batches_in_data(),
        verbose=2
    )

    sed_pred = dnn_output[0] > 0.5
    doa_pred = dnn_output[1]
    sed_pred = evaluation_metrics.reshape_3Dto2D(sed_pred)
    doa_pred = evaluation_metrics.reshape_3Dto2D(doa_pred)

    num_classes = sed_pred.shape[-1]
    num_dims_xyz = 3

    if doa_pred.shape[-1] > num_classes * num_dims_xyz:  # true means we are using masked mse
        sed_mask = np.repeat(sed_pred, 3, -1)
        doa_pred = doa_pred[..., num_classes:] * sed_mask
        doa_gt = doa_gt[..., num_classes:] * sed_mask    

    sed_loss = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt,
                                                     data_gen_test.nb_frames_1s())
    if params['azi_only']:
        doa_loss, conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred,
                                                                           doa_gt,
                                                                           sed_pred,
                                                                           sed_gt)
    else:
        doa_loss, conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred,
                                                                            doa_gt,
                                                                            sed_pred,
                                                                            sed_gt)

    epoch_metric_loss = np.mean([
        sed_loss[0],
        1 - sed_loss[1],
        2 * np.arcsin(doa_loss[1] / 2.0) / np.pi,
        1 - (doa_loss[5] / float(doa_gt.shape[0]))]
    )

    logger.info(
        'F1_overall: %.2f, ER_overall: %.2f, '
        'doa_error_gt: %.2f, doa_error_pred: %.2f, good_pks_ratio:%.2f, '
        'error_metric: %.2f' %
        (
            sed_loss[1], sed_loss[0],
            doa_loss[1], doa_loss[2], doa_loss[5] / float(sed_gt.shape[0]),
            epoch_metric_loss
        )
    )

    logger.info('DOA Metrics: doa_loss_gt: {}, doa_loss_pred: {}, good_pks_ratio: {}'.format(
        doa_loss[1], doa_loss[2], doa_loss[5] / float(sed_gt.shape[0])))
    logger.info(
        'SED Metrics: F1_overall: {}, ER_overall: {}'.format(sed_loss[1], sed_loss[0]))

    logger.info('unique_name: {} '.format(unique_name))


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
        second input: task_id - (optional) To chose the system configuration in parameters.py. 
                                (default) uses default parameters
    
    if len(argv) != 4:
        logger.info('\n\n')
        logger.info('-------------------------------------------------------------------------------------------------------')
        logger.info('The code expected three inputs')
        logger.info('\t>> python seld.py <job-id> <train-test> <task-id>')
        logger.info('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        logger.info('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        logger.info('Using default inputs for now')
        logger.info('-------------------------------------------------------------------------------------------------------')
        logger.info('\n\n')
	"""
    job_id = 1 if len(argv) < 2 else argv[1]

    # use parameter set defined by user
    task_id = '1' if len(argv) < 3 else argv[2]
    params = parameter.get_params(task_id)

    isTraining = True if len(argv) < 4 else (True if argv[3] == 'train' else False)
    logger.info(f"isTraining {isTraining}")

    log_dir_name = None if len(argv) < 5 else argv[4]
    if not log_dir_name and not isTraining:
        raise ValueError("Specify log_dir if evaluation mode")

    model_dir = os.path.join(os.pardir, 'models')
    if isTraining:
        utils.create_folder(model_dir)

    unique_name = '{}_ov{}_train{}_val{}_{}'.format(
        params['dataset'], list_to_string(params['overlap']), list_to_string(params['train_split']),
        list_to_string(params['val_split']),
        job_id)
    
    if not isTraining:
        unique_name = job_id
        
    logger.info(f"unique_name: {unique_name}")

    dnn_type = 'QTCN' if params['use_quaternions'] else params['recurrent_type']
    if not log_dir_name:
        log_dir_name = "-".join([dnn_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")])
    logger.info(f"log_dir_name: {log_dir_name}")

    log_dir = os.path.join(model_dir, unique_name, log_dir_name)
    logger.info(f"log_dir: {log_dir}")

    if isTraining:
        utils.create_folder(log_dir)

    utils.setup_logger(log_dir, console_logger_level=logging.INFO)
    
    logger.info(f"log_dir {log_dir}")
    logger.info("unique_name: {}\n".format(unique_name))
    
    data_gen_train = None
    data_gen_val = None
    data_gen_test = None
    if isTraining:
        load_files_train_splitting_point = None if params['train_val_split'] == 1.0 else 'before'
        data_gen_train = cls_data_generator.DataGenerator(
            dataset=params['dataset'], ov=params['overlap'], split=params['train_split'], db=params['db'],
            nfft=params['nfft'],
            batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
            weakness=params['weakness'], datagen_mode='train', cnn3d=params['cnn_3d'],
            xyz_def_zero=params['xyz_def_zero'],
            azi_only=params['azi_only'], debug_load_single_batch=params['debug_load_single_batch'],
            data_format=params['data_format'], params=params,
            load_files_before_after_splitting_point=load_files_train_splitting_point
        )

        if not params['quick_test']:
            load_files_val_splitting_point = None if params['train_val_split'] == 1.0 else 'after'
            data_gen_val = cls_data_generator.DataGenerator(
                dataset=params['dataset'], ov=params['overlap'], split=params['val_split'], db=params['db'],
                nfft=params['nfft'],
                batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
                weakness=params['weakness'], datagen_mode='train', cnn3d=params['cnn_3d'],
                xyz_def_zero=params['xyz_def_zero'],
                azi_only=params['azi_only'], shuffle=False, debug_load_single_batch=params['debug_load_single_batch'],
                data_format=params['data_format'], params=params,
                load_files_before_after_splitting_point=load_files_val_splitting_point
            )
        else:
            import copy
            data_gen_val = copy.deepcopy(data_gen_train)
            logger.warning(f"Quick test, validation set is a deep copy of training set.")

    else:
        data_gen_test = cls_data_generator.DataGenerator(
            dataset=params['dataset'], ov=params['overlap'], split=params['test_split'], db=params['db'],
            nfft=params['nfft'],
            batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
            weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'],
            xyz_def_zero=params['xyz_def_zero'],
            azi_only=params['azi_only'], shuffle=False, debug_load_single_batch=params['debug_load_single_batch'],
            data_format=params['data_format'], params=params
        )

    data_gen_for_shapes = data_gen_train if isTraining else data_gen_test
    data_in, data_out = data_gen_for_shapes.get_data_sizes()
    logger.info(
        'FEATURES:\n'
        '\tdata_in: {}\n'
        '\tdata_out: {}\n'.format(
            data_in, data_out
        )
    )

    logger.info(
        'MODEL:\n'
        '\tdropout_rate: {}\n'
        '\tCNN: nb_cnn_filt: {}, pool_size{}\n'
        '\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'],
            params['nb_cnn3d_filt'] if params['cnn_3d'] else params['nb_cnn2d_filt'], params['pool_size'],
            params['rnn_size'], params['fnn_size']
        )
    )

    network.set_global_num_classes(params)
    keras.backend.set_image_data_format(params['data_format'])
    logger.info(f"Data format set to {params['data_format']}")
    
    model = None
    if isTraining:
        if params['use_quaternions']:
            assert (params['data_format'] == 'channels_last')

        if params['use_giusenso']:
            assert (params['data_format'] == 'channels_first')
            model = keras_model_giusenso.get_model_giusenso(data_in, data_out, params['dropout_rate'],
                                                            params['nb_cnn2d_filt'],
                                                            params['pool_size'], params['fnn_size'], params['loss_weights'])
        else:
            model = network.get_model(input_shape=data_in, output_shape=data_out, dropout_rate=params['dropout_rate'],
                                      pool_size=params['pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], data_format=params['data_format'],
                                      params=params)
    
    model_path = os.path.join(log_dir, 'model')
    logger.info(f"model_path {model_path}")
    if os.path.exists(model_path):
        logger.info(f"Loading pretrained model from {model_path}")
        model = network.load_seld_model(model_path, params['doa_objective'])
    else:
        if not isTraining:
            raise FileNotFoundError(f"test mode but model was not found at {os.path.abspath(model_path)}")

    try:
        dot_img_file = os.path.join(log_dir, 'model_plot.png')
        keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    except ImportError:
        logger.warning(f"Failed to import pydot, skip plotting")

    if isTraining:
        utils.copy_source_code(log_dir)
        train(model, data_gen_train, data_gen_val, params, log_dir=log_dir, unique_name=unique_name)
    else:
        evaluate(model, data_gen_test, params, log_dir=log_dir, unique_name=unique_name)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

