#
# Data generator for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from collections import deque
import random

import logging
logger = logging.getLogger(__name__)

class DataGenerator(object):
    def __init__(
            self, datagen_mode='train', dataset='ansim', ov=1, split=1, db=30, batch_size=32, seq_len=64,
            shuffle=True, nfft=512, classifier_mode='regr', weakness=0, cnn3d=False, xyz_def_zero=False, extra_name='',
            azi_only=False, debug_load_single_batch=False, data_format='channels_first', params=None,
            load_files_before_after_splitting_point=None
    ):
        if params is None:
            params = {}
        self.params = params
        self._datagen_mode = datagen_mode
        self._classifier_mode = classifier_mode
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(dataset=dataset, ov=ov, split=split, db=db, nfft=nfft)
        self._label_dir = self._feat_cls.get_label_dir(classifier_mode, weakness, extra_name)
        self._feat_dir = self._feat_cls.get_normalized_feat_dir(extra_name)
        self._thickness = weakness
        self._xyz_def_zero = xyz_def_zero
        self._azi_only = azi_only
        self._debug_load_single_batch = debug_load_single_batch
        self._data_format = data_format

        self._nb_frames_file = 0     # Assuming number of frames in feat files are the same
        self._feat_len = None
        self._2_nb_ch = 2 * self._feat_cls.get_nb_channels()
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = len(self._class_dict.keys())
        self._default_azi, self._default_ele = self._feat_cls.get_default_azi_ele_regr()
        self._is_cnn3d_model = cnn3d

        self._filenames_list = []
        self.create_filenames_list(load_files_before_after_splitting_point)

        self.get_feature_label_shapes()

        self._batch_seq_len = self._batch_size*self._seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None

        if self._debug_load_single_batch:
            num_files_for_one_batch = int(np.ceil(float(self._batch_seq_len)/self._nb_frames_file))
            num_files_for_one_batch = max(num_files_for_one_batch, 1)
            self._filenames_list = self._filenames_list[:num_files_for_one_batch]

        self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._batch_seq_len))))
        logger.info(f"Data generator {datagen_mode}: {self._nb_total_batches} batches per epoch.")
        assert (self._nb_total_batches >= 1)

        logger.info(
            'Datagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            'nb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                self._datagen_mode, len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._feat_len, self._2_nb_ch, self._label_len
                )
        )

        logger.info(
            'Dataset: {}, ov: {}, split: {}\n'
            'batch_size: {}, seq_len: {}, shuffle: {}\n'
            'label_dir: {}\n '
            'feat_dir: {}\n'.format(
                dataset, ov, split,
                self._batch_size, self._seq_len, self._shuffle,
                self._label_dir, self._feat_dir
            )
        )

        logger.debug("Complete file list:")
        for file_name in self._filenames_list:
            logger.debug(file_name)

    def get_data_sizes(self):
        if self._data_format == 'channels_first':
            feat_shape = (self._batch_size, self._2_nb_ch, self._seq_len, self._feat_len)
        else:
            feat_shape = (self._batch_size, self._seq_len, self._feat_len, self._2_nb_ch)

        doa_shape = (self._batch_size, self._seq_len, self._nb_classes*(2 if self._azi_only else 3))
        if self.params['doa_objective'] == 'masked_mse':  # add sed ground truth for masking
            doa_shape = (self._batch_size, self._seq_len, self._nb_classes + self._nb_classes*(2 if self._azi_only else 3))

        label_shape = [
            (self._batch_size, self._seq_len, self._nb_classes),
            doa_shape
        ]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def create_filenames_list(self, load_files_before_after_splitting_point_):
        file_list = sorted(os.listdir(self._label_dir))
        if len(file_list) == 0:
            raise FileNotFoundError

        for filename in file_list:
            # if self._datagen_mode in filename:
            self._filenames_list.append(filename)
        if len(self._filenames_list) == 0:
            raise FileNotFoundError

        num_files = len(self._filenames_list)
        logger.info(f"Total number of files {num_files}")
        split_idx = int(num_files*float(self.params['train_val_split']))
        if load_files_before_after_splitting_point_ == 'before':
            self._filenames_list = self._filenames_list[:split_idx]
        elif load_files_before_after_splitting_point_ == 'after':
            self._filenames_list = self._filenames_list[split_idx:]
        logger.info(f"Number of files after splitting {len(self._filenames_list)}")

    def get_feature_label_shapes(self):
        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._feat_len = temp_feat.shape[1] // self._2_nb_ch

        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
        self._label_len = temp_label.shape[-1]
        self._doa_len = (self._label_len - self._nb_classes) // self._nb_classes

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """

        while 1:
            if self._shuffle:
                random.shuffle(self._filenames_list)

            # Ideally this should have been outside the while loop. But while generating the test data we want the data
            # to be the same exactly for all epoch's hence we keep it here.
            self._circ_buf_feat = deque()
            self._circ_buf_label = deque()

            file_cnt = 0

            assert(self._nb_total_batches >= 1)
            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain at least one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)
                        self._circ_buf_label.append(temp_label[row_cnt])
                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._batch_seq_len, self._feat_len * self._2_nb_ch))
                label = np.zeros((self._batch_seq_len, self._label_len))
                for j in range(self._batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                    label[j, :] = self._circ_buf_label.popleft()
                feat = np.reshape(feat, (self._batch_seq_len, self._feat_len, self._2_nb_ch))

                # Split to sequences
                feat = self._split_in_seqs(feat)
                if self._data_format == 'channels_first':
                    feat = np.transpose(feat, (0, 3, 1, 2))
                label = self._split_in_seqs(label)

                if self._azi_only:
                    # Get Cartesian coordinates from azi/ele
                    azi_rad = label[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
                    x = np.cos(azi_rad)
                    y = np.sin(azi_rad)

                    # Set default Cartesian x,y,z coordinates to 0,0,0
                    if self._xyz_def_zero:
                        no_ele_ind = np.where(label[:, :, 2 * self._nb_classes:] == self._default_ele)
                        x[no_ele_ind] = 0
                        y[no_ele_ind] = 0

                    label = [
                        label[:, :, :self._nb_classes],  # SED labels
                        np.concatenate((x, y), -1)       # DOA Cartesian labels
                    ]
                else:
                    # Get Cartesian coordinates from azi/ele
                    azi_rad = label[:, :, self._nb_classes:2 * self._nb_classes] * np.pi / 180
                    ele_rad = label[:, :, 2 * self._nb_classes:] * np.pi / 180
                    tmp_label = np.cos(ele_rad)

                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)

                    # Set default Cartesian x,y,z coordinates to 0,0,0
                    if self._xyz_def_zero:
                        no_ele_ind = np.where(label[:, :, 2 * self._nb_classes:] == self._default_ele)
                        x[no_ele_ind] = 0
                        z[no_ele_ind] = 0
                        y[no_ele_ind] = 0

                    sed_gt = label[:, :, :self._nb_classes]  # SED labels
                    doa_gt = np.concatenate((x, y, z), -1)  # DOA Cartesian labels
                    if self.params['doa_objective'] == 'masked_mse': # add sed ground truth for masking
                        doa_gt = np.concatenate((sed_gt, doa_gt), -1)  # DOA Cartesian labels
                    label = [sed_gt, doa_gt]

                yield feat, label

    def _split_in_seqs(self, data):
        if len(data.shape) == 1:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % self._seq_len:
                data = data[:-(data.shape[0] % self._seq_len), :, :]
            data = data.reshape((data.shape[0] // self._seq_len, self._seq_len, data.shape[1], data.shape[2]))
        else:
            logger.error('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] // num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            logger.error('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_list_index(self, azi, ele):
        return self._feat_cls.get_list_index(azi, ele)

    def get_matrix_index(self, ind):
        return np.array(self._feat_cls.get_vector_index(ind))

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()
