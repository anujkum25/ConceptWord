from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


__path__ = os.path.abspath(os.path.dirname(__file__))
LSMDC_DATA_PATH = os.path.normpath(os.path.join(__path__, "../dataset/LSMDC"))


class ModelConfig(object):

    def __init__(self):
        self.batch_size = 32

        self.video_height = 299
        self.video_width = 299
        self.video_steps = 40
        self.caption_length = 40

        self.attr_length = 10

        self.word_dim = 500
        self.num_layers = 2
        self.hidden_dim = 500
        self.video_embedding_size = 500

        self.image_feature_net = 'resnet'
        self.layer = 'res5c'

        self.dropout_keep_prob = 0.8
        self.lstm_cell = 'BasicLSTM'
        self.loss_weight = 0.1

        self.concept_detector = "TracingLSTM"
        self.attention_method = "SemanticAttention"


class TrainConfig(object):

    def __init__(self):
        self.learning_rate = 0.0001
        self.train_dir = None
        self.max_steps = 1000000

        self.num_epochs = 20

        self.learning_rate_decay_steps = 50000
        self.learning_rate_decay_factor = 0.5
        self.optimizer = 'Adam'
        self.max_grad_norm = 30.0

        self.steps_per_logging = 4
        self.steps_per_evaluate = 5000
        self.train_tag = 'FIB'
