from labels import labels
import numpy as np
import h5py
import os
import hickle as hkl
import data_util
from videocap.util import log


def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)


class AttributeExtractor:

    def __init__(self,
                 dataset=None,
                 attr_length=20,
                 top_num=5):
        self.attr_length = attr_length
        self.dataset = dataset
        self.top_num = top_num


    def __del__(self):
        self.feat_hdf5.close()
        self.tgif_hdf5.close()

    def read_word_vocabulary(self):
        if self.dataset and hasattr(self.dataset, 'idx2word'):
            self.idx2word = self.dataset.idx2word
            self.word2idx = self.dataset.word2idx
            self.word_matrix = self.dataset.word2idx
        else:
            word_matrix_path = os.path.join(VOCABULARY_DIR, 'word_matrix.hkl')
            assert_exists(word_matrix_path)
            word2idx_path = os.path.join(VOCABULARY_DIR, 'word_to_index.hkl')
            assert_exists(word2idx_path)
            idx2word_path = os.path.join(VOCABULARY_DIR, 'index_to_word.hkl')
            assert_exists(idx2word_path)

            with open(word_matrix_path, 'r') as f:
                self.word_matrix = hkl.load(f)

            with open(word2idx_path, 'r') as f:
                self.word2idx = hkl.load(f)

            with open(idx2word_path, 'r') as f:
                self.idx2word = hkl.load(f)

    def read_tgif_file(self):
        f = h5py.File(TGIF_PATH, 'r')
        log.info("Load TGIF attribute : {}".format(TGIF_PATH))
        return f

    def read_hdf5_file(self):
        f = h5py.File(HDF5_PATH, 'r')
        log.info("Load attributes : {}".format(HDF5_PATH))
        return f

    def read_msr_file(self):
        f = h5py.File(MSR_PATH, 'r')
        log.info("Load MSR attribute : {}".format(MSR_PATH))
        return f

    def read_crc_file(self):
        f = h5py.File(CRC_PATH, 'r')
        log.info("Load CRC attribute : {}".format(CRC_PATH))
        return f

    def transit_to_vocab_idx(self, imagenet_idx):
        target = labels[imagenet_idx]
        target = [t.strip() for t in target.split(',')]
        target = target[0]
        target = data_util.clean_str(target)
        try:
            target = self.word2idx[target.split()[-1]]
        except:
            target = 0
        return target

    def get_top_attribute(self, video_id):
        if video_id[:4] == 'tgif':
            attributes = list(self.tgif_hdf5[video_id])
        elif video_id[:3] == 'vid':
            attributes = list(self.feat_hdf5[video_id])
        elif video_id[:3] == 'vas':
            attributes = list(self.crc_hdf5[video_id])
        elif video_id[:3] == 'msr':
            attributes = list(self.msr_hdf5[video_id])
        else:
            raise Exception("wrong video_id")

        att_indices = [self.word2idx[att] for att in attributes if att in self.word2idx.keys()]
        padded_indices = data_util.pad_sequences(sequences=[att_indices],
                                                 pad_token=0,
                                                 pad_location="RIGHT",
                                                 max_length=self.attr_length)
        mask_indices = data_util.fill_mask(max_length=self.attr_length,
                                           current_length=len(att_indices),
                                           zero_location="RIGHT")
        return (np.array(padded_indices[0]), mask_indices)
