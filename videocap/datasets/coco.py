import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from videocap.util import log

import itertools
import re
import os.path
import random
import h5py

__path__ = os.path.abspath(os.path.dirname(__file__))

def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

COCO_DATA_DIR = os.path.normpath(os.path.join(__path__, '../../dataset/mscoco'))
assert_exists(COCO_DATA_DIR)

RE_ALPHABET_SPACE = re.compile('[^a-zA-Z ]')


class DatasetMSCOCO():
    '''
    Access API for MS-COCO dataset.
    '''

    def __init__(self,
                 dataset_name = 'val2014',
                 image_feature_net = 'vgg19',
                 max_n_images = None,
                 ):
        self.dataset_name = dataset_name
        self.image_feature_net = image_feature_net

        caption_file = os.path.join(COCO_DATA_DIR,
                                    'annotations/captions_%s.json' % dataset_name)
        assert_exists(caption_file)
        log.info("Using COCO caption file : %s", caption_file)
        self.coco = COCO(caption_file)

        self.image_ids = list(self.coco.getImgIds())
        if max_n_images is not None:
            log.debug("Using only %d images out of total %d images (%s)",
                      max_n_images, len(self.image_ids), self.dataset_name)
            self.image_ids = self.image_ids[:max_n_images] # TODO shuffle

        feature_file = os.path.join(COCO_DATA_DIR,
                                    'features/%s.%s.h5' % (dataset_name, image_feature_net))
        assert_exists(feature_file)
        self.feat_h5 = h5py.File(feature_file, 'r')

    def __del__(self):
        self.feat_h5.close()

    def __len__(self):
        ''' The number of images in this dataset. '''
        return len(self.image_ids)

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<DatasetMSCOCO (%s) with %d images and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<DatasetMSCOCO (%s) with %d images -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        '''
        words = RE_ALPHABET_SPACE.sub('', sentence).lower().split()
        for w in words:
            if not w: continue
            yield w

    def build_word_vocabulary(self,
                              all_captions_source=None,
                              word_count_threshold=0,
                              ):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''
        log.infov('Building word vocabulary (%s) ...', self.dataset_name)

        if all_captions_source is None:
            all_captions_source = self.iter_all_captions()

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        for sentence in tqdm(list(all_captions_source),
                             desc='Iterating all sentences'):
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1

        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        log.info("Filtered vocab words (threshold = %d), from %d to %d",
                 word_count_threshold, len(word_counts), len(vocab))

        # build index and vocabularies
        self.word2idx = {}
        self.idx2word = {}

        self.idx2word[0] = '.'
        self.word2idx['#START#'] = 0
        for idx, w in enumerate(vocab, start=1):
            self.word2idx[w] = idx
            self.idx2word[idx] = w

        word_counts['.'] = nsents
        bias_init_vector = np.array([1.0*word_counts[w] for i, w in self.idx2word.iteritems()])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        self.bias_init_vector = bias_init_vector

    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(dataset, 'word2idx'), \
            'The dataset instance should have idx2word and word2idx'
        assert isinstance(dataset.idx2word, dict) and isinstance(dataset.word2idx, dict), \
            'The dataset instance should have idx2word and word2idx (as dict)'

        if hasattr(self, 'word2idx'):
            log.warn("Overriding %s' word vocabulary from %s ...", self, dataset)

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx



    # Dataset Access APIs (batch loading, etc)
    # ========================================

    def iter_image_ids(self, shuffle=False):
        '''
        Iterate image ids in integers (e.g. 262148, ...)
        '''
        ids = list(self.image_ids)    # make sure to create a copy of list
        if shuffle:
            random.shuffle(ids)
        for id in ids:
            yield id

    def iter_image_filenames(self, shuffle=False):
        '''
        Iterate image filenames in strings
        (e.g. 'COCO_val2014_000000262148.jpg', ...)
        '''
        for image_id in self.iter_image_ids(shuffle=shuffle):
            image_object = self.coco.imgs[image_id]
            yield image_object['file_name']

    def iter_all_captions(self):
        '''
        Iterate caption strings associated in the images.
        '''
        for id in self.iter_image_ids():
            for cap in self.iter_captions_for_image(id):
                yield cap

    def iter_captions_for_image(self, image_id):
        '''
        Iterate caption strings for the given image id.
        '''
        assert isinstance(image_id, (int, long))
        for ann in self.coco.imgToAnns[image_id]:
            yield ann['caption']

    def iter_image_caption_pairs(self, shuffle=False):
        '''
        Iterate all (image_id, caption str) pairs in the dataset.
        '''
        for image_id in self.iter_image_ids(shuffle=shuffle):
            for caption in self.iter_captions_for_image(image_id):
                yield (image_id, caption)


    def load_image_feature(self, image_id, feature_name='fc7'):
        assert feature_name in ['fc7', 'conv5_4']    # TODO: VGG19 only

        image_filename = self.coco.imgs[image_id]['file_name']
        return self.feat_h5[image_filename][feature_name]#.value

    def get_image_feature_dimension(self, feature_name='fc7'):
        assert feature_name in ['fc7', 'conv5_4']    # TODO: VGG19 only
        if feature_name == 'fc7': return (4096, )
        elif feature_name == 'conv5_4': return (14, 14, 512)
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
        return [self.word2idx[w] for w in \
                self.split_sentence_into_words(sentence) \
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
            eos_position = T

            sent = ' '.join(self.idx2word[i] for i in word_matrix[b, :eos_position])
            sentences[b] = sent

        return sentences


    def next_batch(self, batch_size=64, max_length=None,
                   include_extra=False):
        '''
        Prepare the next batch (as dict), which consists of:
            - image ids
            - image features

        Args:
            batch_size: the mini-batch size
            max_length: the maximum of sentence words. Longer sentences
                (in terms of word count) than this will be stripped.
            include_extra: whether to include additional debug information
                such as raw sentences, etc. Defaults to False (i.e. exclude)
                for run-time performance concerns.
        '''
        if not hasattr(self, '_batch_it'):
            self._batch_it = itertools.cycle(self.iter_image_caption_pairs())

        # first, collect the current batch chunk
        chunk = []
        for k in xrange(batch_size):
            image_id, sentence = next(self._batch_it)
            chunk.append((image_id, sentence))

        # maximum sentence length. if None, max{ #words in sentence } is used
        # TODO check RNN max length and <eos>
        if max_length is None:
            max_length = max(len(list(self.split_sentence_into_words(s))) \
                             for _, s in chunk)

        # fill up the batch tensors
        batch_image_feature_convmap = np.zeros([batch_size] + list(self.get_image_feature_dimension('conv5_4')), dtype=np.float32)
        batch_image_id = np.zeros([batch_size], dtype=np.uint32)
        batch_caption_words = np.zeros([batch_size, max_length], dtype=np.uint32)
        #batch_caption_length = np.zeros([batch_size], dtype=np.uint32)

        if include_extra:
            batch_caption_sentence = np.asarray([None] * batch_size)

        for k in xrange(batch_size):
            image_id, sentence = chunk[k]

            # (i) image id and features (e.g. fc7)
            batch_image_id[k] = image_id
            batch_image_feature_convmap[k, :] = self.load_image_feature(image_id, 'conv5_4')

            # (ii) sentence and its words representation
            if include_extra:
                batch_caption_sentence[k] = sentence

            # sentence -> word list -> word matrix (sequence padding) and mask
            sentence_word_indices = self.convert_sentence_to_matrix(sentence)
            T = len(sentence_word_indices)
            #print sentence, sentence_word_indices

            # TODO handle <EOS> and dot ('.') properly
            length = min(T, max_length)
            batch_caption_words[k, :length] = sentence_word_indices
            #batch_caption_length[k] = length


        ret = {
            'image_id' : batch_image_id,
            'image_feature_convmap' : batch_image_feature_convmap,
            'caption_words' : batch_caption_words,
            #'caption_length' : batch_caption_length,
        }
        if include_extra:
            ret['caption_sentence'] = batch_caption_sentence

        return ret
