import numpy as np

from videocap.util import log
from attribute_extractor import AttributeExtractor

import itertools
import re
import os.path
import random
import h5py

import pandas as pd
import data_util
import hickle as hkl

# For debug purpose
import pudb

__path__ = os.path.abspath(os.path.dirname(__file__))
eos_word = '<EOS>'


def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

# FIXME
LSMDC_DATA_DIR = os.path.normpath(os.path.join(__path__, '../../dataset/LSMDC'))
assert_exists(LSMDC_DATA_DIR)

DATAFRAME_DIR = os.path.join(LSMDC_DATA_DIR, 'DataFrame')
assert_exists(DATAFRAME_DIR)

VOCABULARY_DIR = os.path.join(LSMDC_DATA_DIR, 'Vocabulary')
assert_exists(VOCABULARY_DIR)

VIDEO_FEATURE_DIR = os.path.join(LSMDC_DATA_DIR, 'LSMDC16_features')
assert_exists(VIDEO_FEATURE_DIR)


class DatasetLSMDC():
    '''
    Access API for LSMDC Videos.
    '''

    def __init__(self,
                 dataset_name='train',
                 image_feature_net='resnet',
                 layer='pool5',
                 padding=True,
                 max_length=80,
                 use_tgif=False,
                 max_n_videos=None,
                 attr_length=20,
                 top_num=5
                 ):
        self.use_tgif = use_tgif
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.image_feature_net = image_feature_net.lower()
        self.layer = layer
        self.data_df = self.read_df_from_csvfile()
        self.attr_length = attr_length
        self.top_num = top_num

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]

        self.video_ids = self.get_video_ids()
        self.feat_h5 = self.read_feat_from_hdf5()
        if self.use_tgif:
            self.tgif_h5 = self.read_tgif_hdf5()

    def __del__(self):
        self.feat_h5.close()
        self.tgif_h5.close()

    def __len__(self):
        ''' The number of images in this dataset. '''
        return len(self.video_ids)

    def get_video_ids(self):
        return list(self.data_df.index)

    def read_tgif_hdf5(self):
        if self.image_feature_net.lower() == 'resnet':
            if self.layer.lower() == 'res5c':
                feature_file = os.path.join(VIDEO_FEATURE_DIR, 'TGIF_' + self.image_feature_net.upper()+".hdf5")
                assert_exists(feature_file)
            elif self.layer.lower() == 'pool5':
                feature_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_" + self.image_feature_net.upper()
                                            + "_" + self.layer.lower() + ".hdf5")
                assert_exists(feature_file)
            elif self.layer.lower() == 'fc1000':
                feature_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_" + self.image_feature_net.upper()
                                            + "_" + self.layer.lower() + ".hdf5")
                assert_exists(feature_file)
        elif self.image_feature_net.lower() == 'c3d':
            feature_file = os.path.join(VIDEO_FEATURE_DIR, "TGIF_" + self.image_feature_net.upper() + ".hdf5")
            assert_exists(feature_file)

        log.info("Load %s hdf5 file: %s", self.image_feature_net.upper(), feature_file)
        return h5py.File(feature_file, 'r')

    def read_feat_from_hdf5(self):
        if self.image_feature_net.lower() == 'resnet':
            if self.layer.lower() == 'res5c':
                feature_file = os.path.join(VIDEO_FEATURE_DIR, self.image_feature_net.upper()+".hdf5")
                assert_exists(feature_file)
            elif self.layer.lower() == 'pool5':
                feature_file = os.path.join(VIDEO_FEATURE_DIR, self.image_feature_net.upper()
                                            + "_" + self.layer.lower() + ".hdf5")
                assert_exists(feature_file)
            elif self.layer.lower() == 'fc100':
                feature_file = os.path.join(VIDEO_FEATURE_DIR, self.image_feature_net.upper()
                                            + "_" + self.layer.lower() + ".hdf5")
                assert_exists(feature_file)
        elif self.image_feature_net.lower() == 'c3d':
            feature_file = os.path.join(VIDEO_FEATURE_DIR, self.image_feature_net.upper() + ".hdf5")
            assert_exists(feature_file)
        elif self.image_feature_net.lower() == 'google':
            feature_file = os.path.join(VIDEO_FEATURE_DIR, self.image_feature_net.upper() + ".hdf5")
            assert_exists(feature_file)

        log.info("Load %s hdf5 file : %s", self.image_feature_net.upper(), feature_file)

        return h5py.File(feature_file, 'r')

    def read_df_from_csvfile(self):
        if self.dataset_name == 'train':
            data_df_path = os.path.join(DATAFRAME_DIR, 'LSMDC16_CAP_train.csv')
            assert_exists(data_df_path)
        elif self.dataset_name == 'validation':
            data_df_path = os.path.join(DATAFRAME_DIR, 'LSMDC16_CAP_val.csv')
            assert_exists(data_df_path)
        elif self.dataset_name == 'test':
            data_df_path = os.path.join(DATAFRAME_DIR, 'LSMDC16_CAP_test.csv')
            assert_exists(data_df_path)

        data_df = pd.read_csv(data_df_path, sep='\t')
        log.info("Load %s csv file : %s", self.dataset_name, data_df_path)

        data_df = data_df.set_index('key')

        extract_field = ['description']
        if self.use_tgif:
            tgif_df_path = os.path.join(DATAFRAME_DIR, 'TGIF.csv')
            assert_exists(tgif_df_path)
            tgif_df = pd.read_csv(tgif_df_path, sep='\t')
            tgif_df = tgif_df.set_index('key')
            log.info("Load %s csv file : %s", 'TGIF', tgif_df_path)
            return pd.concat([data_df.loc[:, extract_field], tgif_df.loc[:, extract_field]], axis=0)
        else:
            return data_df.loc[:, extract_field]

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<DatasetLSMDC (%s) with %d videos and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<DatasetLSMDC (%s) with %d videos -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        words = data_util.clean_str(sentence).split()
        words = words + [eos_word]
        for w in words:
            if not w:
                continue
            yield w

    def build_word_vocabulary(self):
        word_matrix_path = os.path.join(VOCABULARY_DIR, 'word_matrix.hkl')
        assert_exists(word_matrix_path)
        word2idx_path = os.path.join(VOCABULARY_DIR, 'word_to_index.hkl')
        assert_exists(word2idx_path)
        idx2word_path = os.path.join(VOCABULARY_DIR, 'index_to_word.hkl')
        assert_exists(idx2word_path)

        with open(word_matrix_path, 'r') as f:
            self.word_matrix = hkl.load(f)
        log.info("Load word_matrix from hkl file : %s", word_matrix_path)

        # TODO word2idx file should be dict. now it is list.
        with open(word2idx_path, 'r') as f:
            self.word2idx = hkl.load(f)
        log.info("Load word2idx from hkl file : %s", word2idx_path)

        with open(idx2word_path, 'r') as f:
            self.idx2word = hkl.load(f)
        log.info("Load idx2word from hkl file : %s", idx2word_path)

        self.attribute_extractor = AttributeExtractor(dataset=self,
                                                      attr_length=self.attr_length,
                                                      top_num=self.top_num)
#    def build_word_vocabulary(self,
#                              all_captions_source=None,
#                              word_count_threshold=0,
#                              ):
#        '''
#        borrowed this implementation from @karpathy's neuraltalk.
#        '''
#        log.infov('Building word vocabulary (%s) ...', self.dataset_name)
#
#        if all_captions_source is None:
#            all_captions_source = self.iter_all_captions()
#
#        # enumerate all sentences to build frequency table
#        word_counts = {}
#        nsents = 0
#        for sentence in tqdm(list(all_captions_source),
#                             desc='Iterating all sentences'):
#            nsents += 1
#            for w in self.split_sentence_into_words(sentence):
#                word_counts[w] = word_counts.get(w, 0) + 1
#
#        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
#        log.info("Filtered vocab words (threshold = %d), from %d to %d",
#                 word_count_threshold, len(word_counts), len(vocab))
#
#        # build index and vocabularies
#        self.word2idx = {}
#        self.idx2word = {}
#
#        self.idx2word[0] = '.'
#        self.word2idx['#START#'] = 0
#        for idx, w in enumerate(vocab, start=1):
#            self.word2idx[w] = idx
#            self.idx2word[idx] = w
#
#        word_counts['.'] = nsents
#        bias_init_vector = np.array([1.0*word_counts[w] for i, w in self.idx2word.iteritems()])
#        bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
#        bias_init_vector = np.log(bias_init_vector)
#        bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range
#        self.bias_init_vector = bias_init_vector

    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(dataset, 'word2idx'), \
            'The dataset instance should have idx2word and word2idx'
        assert (isinstance(dataset.idx2word, dict) or isinstance(dataset.idx2word, list)) \
                and isinstance(dataset.word2idx, dict), \
            'The dataset instance should have idx2word and word2idx (as dict)'

        if hasattr(self, 'word2idx'):
            log.warn("Overriding %s' word vocabulary from %s ...", self, dataset)

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx
        if hasattr(dataset, 'word_matrix'):
            self.word_matrix = dataset.word_matrix

        self.attribute_extractor = AttributeExtractor(dataset=self,
                                                      attr_length=self.attr_length,
                                                      top_num=self.top_num)

    # Dataset Access APIs (batch loading, etc)
    # ========================================

    def iter_video_ids(self, shuffle=False):
        '''
        Iterate video ids. (e.g. vid98834, ...)
        '''
        ids = list(self.video_ids)
        if shuffle:
            random.shuffle(ids)
        for id in ids:
            yield id

    def iter_all_captions(self):
        '''
        Iterate caption strings associated in the images.
        '''
        for id in self.iter_video_ids():
            yield self.get_captions_for_video(id)

    def get_root_for_video(self, video_id):
        '''
        Return root for description of video given id
        '''
        return self.data_df.loc[video_id, 'root']

    def get_captions_for_video(self, video_id):
        '''
        Return caption strings for the given video id.
        '''
        return self.data_df.loc[video_id, 'description']

    def iter_video_caption_root_pairs(self, shuffle=False):
        '''
        Iterate all (video_id, caption str, root) pairs in the dataset.
        '''
        for video_id in self.iter_video_ids(shuffle=shuffle):
            yield (video_id,
                   self.get_captions_for_video(video_id))

    def load_video_feature(self, video_id):
        if video_id[:4] == 'tgif':
            video_feature = np.array(self.tgif_h5[video_id])
        elif video_id[:3] == 'vid':
            video_feature = np.array(self.feat_h5[video_id])
        else:
            raise Exception('video_key error in load_video_feature')

        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['fc1000', 'pool5', 'res5c']
            if self.layer.lower() == 'res5c':
                video_feature = np.transpose(video_feature, [0, 2, 3, 1])
                assert list(video_feature.shape[1:]) == [7, 7, 2048]
            elif self.layer.lower() == 'pool5':
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 2048]
            elif self.layer.lower() == 'fc1000':
                video_feature = np.expand_dims(video_feature, axis=1)
                video_feature = np.expand_dims(video_feature, axis=1)
                assert list(video_feature.shape[1:]) == [1, 1, 1000]
        elif self.image_feature_net.lower() == 'c3d':
            assert list(video_feature.shape) == [20, 4096]
            video_feature = np.expand_dims(video_feature, axis=1)
            video_feature = np.expand_dims(video_feature, axis=1)
            assert list(video_feature.shape[1:]) == [1, 1, 4096]
        elif self.image_feature_net.lower() == 'google':
            assert list(video_feature.shape) == [80, 1024]
            video_feature = np.expand_dims(video_feature, axis=1)
            video_feature = np.expand_dims(video_feature, axis=1)
            assert list(video_feature.shape[1:]) == [1, 1, 1024]

        return video_feature

    def get_video_feature_dimension(self):
        if self.image_feature_net == 'resnet':
            assert self.layer.lower() in ['fc1000', 'pool5', 'res5c']
            if self.layer.lower() == 'res5c':
                return (self.max_length, 7, 7, 2048)
            elif self.layer.lower() == 'pool5':
                return (self.max_length, 1, 1, 2048)
            elif self.layer.lower() == 'fc1000':
                return (self.max_length, 1, 1, 1000)
        elif self.image_feature_net.lower() == 'c3d':
            return (self.max_length, 1, 1, 4096)
        elif self.image_feature_net.lower() == 'google':
            return (self.max_length, 1, 1, 1024)
        raise NotImplementedError()

    def convert_sentence_to_matrix(self, sentence):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.

        Args:
            sentence: A str for unnormalized sentence, containing T words

        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        return [self.word2idx[w] for w in
                self.split_sentence_into_words(sentence)
                if w in self.word2idx]

    def assemble_into_sentence(self, word_matrix):
        '''
        Convert the word matrix (Batch x MaxLength) into a list of
        human-readable setnences, w.r.t the current directory.
        '''
        B, T = word_matrix.shape
        sentences = [None] * B

        for b in xrange(B):
            # TODO get eos position
            if 2 in word_matrix[b]:
                eos_position = list(word_matrix[b]).index(2)
            else:
                eos_position = len(word_matrix[b])

            sent = ' '.join(self.idx2word[int(i)] for i in word_matrix[b, :eos_position])
            sentences[b] = sent

        return sentences

    def get_nearest_verb_from_emb(self, verbemb_list):
        '''
        Get nearest verb in dataset
        '''
        B, T = verbemb_list.shape
        verbs = [None] * B

        for b in xrange(B):
            np.sum((self.word_matrix - verbemb_list[b])**2, axis=1)
            nearest_ind = np.argsort(verbemb_list[b])[0]
            verbs[b] = self.idx2word[nearest_ind]

        return verbs

    def get_nearest_verb(self, verb_list):
        '''
        Get nearest verb in dataset
        '''
        B = len(verb_list)
        verbs = [None] * B

        for b in xrange(B):
            verbs[b] = self.idx2word[verb_list[b]]

        return verbs


    def next_batch(self, batch_size=64, include_extra=False, shuffle=True):
        '''
        Prepare the next batch (as dict), which consists of:
            - video ids
            - video features
            - sentence matrices
            - roots

        Args:
            batch_size: the mini-batch size
            include_extra: whether to include additional debug information
                such as raw sentences, etc. Defaults to False (i.e. exclude)
                for run-time performance concerns.
            shuffle : shuffling index of video ids
        '''
        if not hasattr(self, '_batch_it'):
            self._batch_it = \
                itertools.cycle(self.iter_video_caption_root_pairs(shuffle=shuffle))

        # first, collect the current batch chunk
        chunk = []
        for k in xrange(batch_size):
            video_id, sentence = next(self._batch_it)
            chunk.append((video_id, sentence))

        # maximum sentence length. if None, max{ #words in sentence } is used
        # TODO check RNN max length and <eos>
        if self.max_length is None:
            self.max_length = max(len(list(self.split_sentence_into_words(s)))
                                  for _, s, _ in chunk)

        # fill up the batch tensors
        # TODO When using resnet, we can only use layer res5c
        batch_video_feature_convmap = np.zeros([batch_size]
                                               + list(self.get_video_feature_dimension()), dtype=np.float32)
        batch_video_ids = []
        batch_caption_words = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_attribute = np.zeros([batch_size, self.attr_length], dtype=np.uint32)

        # Mask Init
        batch_video_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_caption_mask = np.zeros([batch_size, self.max_length], dtype=np.uint32)
        batch_attribute_mask = np.zeros([batch_size, self.attr_length], dtype=np.uint32)

        if include_extra:
            batch_caption_sentence = np.asarray([None] * batch_size)

        # TODO have to divide role by four function [(i), (ii), (iii), (iv)]
        for k in xrange(batch_size):
            video_id, sentence = chunk[k]

            current_video_feature = self.load_video_feature(video_id)  # [video_lenth, 7, 7, 2048]

            # (i) video id and features (e.g. res5c)
            batch_video_ids.append(video_id)
            # Fill pad to have same length using ``data_util.pad_video``
            batch_video_feature_convmap[k, :] = data_util.pad_video(current_video_feature,
                                                                    self.get_video_feature_dimension())

            # (ii) sentence and its words representation
            if include_extra:
                batch_caption_sentence[k] = sentence

            # sentence -> word list -> word matrix (sequence padding) and mask
            sentence_word_indices = self.convert_sentence_to_matrix(sentence)
            T = len(sentence_word_indices)
            # print sentence, entence_word_indices

            # TODO handle <EOS> and dot ('.') properly
            length = min(T, self.max_length)
            batch_caption_words[k, :length] = sentence_word_indices[:length]
            # batch_caption_length[k] = length

#            # (iii) root representation
#            root = data_util.clean_root(root)
#            if root in self.word2idx.keys():
#                root_index = self.word2idx[root]
#            else:
#                root_index = 0
#            batch_roots[k] = root_index

            # (iv) Build video mask.
            video_length = current_video_feature.shape[0]
            batch_video_mask[k] = data_util.fill_mask(self.max_length,
                                                      video_length,
                                                      zero_location='LEFT')
            batch_caption_mask[k] = data_util.fill_mask(self.max_length,
                                                        len(sentence_word_indices),
                                                        zero_location='RIGHT')

            # (v) Build attribute and attribute_mask
            attr, attr_mask = self.attribute_extractor.get_top_attribute(video_id)
            batch_attribute[k] = attr
            batch_attribute_mask[k] = attr_mask

        ret = {
            'ids': batch_video_ids,
            'video_features': batch_video_feature_convmap,
            'caption_words': batch_caption_words,
            'video_mask': batch_video_mask,
            'caption_mask': batch_caption_mask,
            'attribute': batch_attribute,
            'attribute_mask': batch_attribute_mask
        }
        if include_extra:
            ret['caption_sentence'] = batch_caption_sentence

        return ret

    def batch_iter(self, num_epochs, batch_size):
        for epoch in xrange(num_epochs):
            steps_in_epoch = int(len(self) / batch_size)

            for s in range(steps_in_epoch):
                yield self.next_batch(batch_size,
                                      include_extra=True)  # For DEBUG
