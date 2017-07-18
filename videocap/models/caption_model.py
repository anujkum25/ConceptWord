from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import time

from videocap.models.tracing_lstm import TracingLSTM
from videocap.models.semantic_attention import SemanticAttention
from videocap.util import log
from videocap.datasets import data_util
from videocap import metrics

import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn_cell


_lstm_map = {
    'BasicLSTM': rnn_cell.BasicLSTMCell,
}
_concept_detector_map = {
    "TracingLSTM": TracingLSTM
}
_attention_method_map = {
    "SemanticAttention": SemanticAttention
}


class CAPGenerator(object):
    """
    """
    def __init__(self, config, word_embed, attr_idx):
        self.config = config
        self.batch_size = config.batch_size
        self.word_embed = word_embed
        self.vocab_size = word_embed.shape[0]
        self.name = 'CaptionGenerator'
        self.attr_idx = attr_idx

        self.dropout_keep_prob = tf.placeholder_with_default(
            self.config.dropout_keep_prob, [])

        self.batch_size = config.batch_size
        self.video_steps = config.video_steps
        self.initializer = tf.contrib.layers.xavier_initializer(
            uniform=False)
        self.video_cell = _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=False)
        self.attr_cell = _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=False)
        self.caption_cell = _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=False)

        self.concept_detector = \
            _concept_detector_map[self.config.concept_detector](config=config, attr_idx=attr_idx)
        self.attention_method = \
            _attention_method_map[self.config.attention_method](config=config)

    def get_feed_dict(self, batch_chunk):
        feed_dict = {
            self.video: batch_chunk['video_features'].astype(float),
            self.video_mask: batch_chunk['video_mask'].astype(float),
            self.caption: batch_chunk['caption_words'].astype(float),
            self.caption_mask: batch_chunk['caption_mask'].astype(float),
            self.target_attribute: batch_chunk['bow'].astype(float),
        }
        return feed_dict

    def get_placeholder(self):
        video = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps, 7, 7, 2048])
        video_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps])
        caption = tf.placeholder(tf.int32, [self.config.batch_size, self.config.caption_length])
        caption_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.caption_length])
        target_attribute = tf.placeholder(tf.float32, [self.config.batch_size, len(self.attr_idx)])
        train_flag = tf.placeholder(tf.bool)

        result = {
            'video': video,
            'video_mask': video_mask,
            'caption': caption,
            'caption_mask': caption_mask,
            'target_attribute': target_attribute,
            'train_flag': train_flag
        }
        return result

    def build_caption_embedding(self, input_seqs, name=None, reuse_variable=False):
        """Builds the input sequence(caption) embeddings.

        Inputs:
            self.input_seqs

        Outputs:
            seq_embeddings
        """
        with tf.variable_scope("seq_embedding", reuse=reuse_variable), tf.device("/cpu:0"):
            seq_embeddings = tf.nn.embedding_lookup(self.word_embed_t, input_seqs, name=name)

        return seq_embeddings

    def build_video_embedding(self, video_cell, video, video_mask, reuse_variable):
        """Builds the video input embeddings.

        Inputs:
            self.video

        Outputs:
            video_embeddings
        """
        batch_size, time_steps, width, height, vid_dim = video.get_shape().as_list()

        with tf.variable_scope("video_rnn", initializer=self.initializer, reuse=reuse_variable) as scope:
            video_pool = tf.reduce_mean(video, [2, 3])
            video_pool_drop = tf.nn.dropout(video_pool, self.dropout_keep_prob)
            vid_initial_state = tf.zeros([batch_size, video_cell.state_size])
            vid_rnn_states = [vid_initial_state]
            for i in range(self.video_steps):
                if i > 0:
                    scope.reuse_variables()
                output, state = video_cell(video_pool_drop[:, i, :],
                                           vid_rnn_states[-1])
                vid_rnn_states.append(state * tf.expand_dims(video_mask[:, i], 1))
        return output, vid_rnn_states[-1]

    def build_caption_decoder(self,
                              caption_cell,
                              vid_emb_state,
                              attributes,
                              reuse_variable=False):

        batch_size = attributes.get_shape().as_list()[0]
        embed_attributes = self.build_caption_embedding(attributes, name="embedding_attributes",
                                                        reuse_variable=True)

        cap_initial_state = vid_emb_state
        cap_rnn_states = [cap_initial_state]
        cap_predicted_words = []
        y_list = [tf.ones([batch_size], dtype=tf.int32)]

        caption_loss = 0.0

        with tf.variable_scope("caption_rnn", reuse=reuse_variable) as scope:
            for i in range(self.config.video_steps):
                if i > 0:
                    scope.reuse_variables()
                word_embed = self.build_caption_embedding(y_list[-1])
                current_x, gammas = self.attention_method.attended_input(word_embed, embed_attributes)
                new_output, new_state = caption_cell(current_x, cap_rnn_states[-1])
                cap_rnn_states.append(new_state)

                current_p, betas = self.attention_method.attended_output(new_output, embed_attributes)
                current_p = tf.nn.dropout(current_p, self.dropout_keep_prob)
                logit_words = slim.fully_connected(current_p, self.vocab_size,
                                                   activation_fn=None, scope=scope, reuse=(i > 0))

                predicted_word = tf.argmax(logit_words, 1)
                cap_predicted_words.append(predicted_word)

                cur_y_if_train = self.caption[:, i]
                cur_y_if_val = tf.cast(predicted_word, tf.int32)
                y_list.append(tf.cond(self.train_flag, lambda: cur_y_if_train, lambda: cur_y_if_val))

                with tf.variable_scope("caption_loss", reuse=reuse_variable):
                    labels = tf.expand_dims(self.caption[:, i], 1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    labels_with_index = tf.concat(axis=1, values=[indices, labels])
                    onehot_labels = tf.sparse_to_dense(labels_with_index,
                                                       tf.stack([batch_size, self.vocab_size]),
                                                       sparse_values=1.0,
                                                       default_value=0)
                    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                    masked_cross_loss = cross_entropy_loss * self.caption_mask[:, i]
                    loss = tf.reduce_sum(masked_cross_loss)
                    caption_loss += loss

        return caption_loss, cap_predicted_words


    def concept_loss(self, logit, target_attrs):
        batch_size = logit.get_shape().as_list()[0]

        targets = tf.reshape(target_attrs, [-1])
        logits = tf.reshape(logit, [-1])
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=targets,
                                                         logits=logits)

        losses = self.config.loss_weight * losses
        batch_loss = tf.div(tf.reduce_sum(losses), batch_size,
                            name='concept_batch_loss')
        tf.contrib.losses.add_loss(batch_loss)

        return batch_loss

    def build_model(self,
                    video,
                    video_mask,
                    caption,
                    caption_mask,
                    target_attribute,
                    train_flag,
                    reuse_variable=False):
        """Builds the model.
        Args:
            videos: A float32 Tensor with shape [batch_size, video_steps, height, width, channels].
            video_masks: An int32 0/1 Tensor with shape [batch_size, video_length]
            input_seqs: An int32 Tensor with shape [batch_size, padded_length].
            target_seqs: An int32 Tensor with shape [batch_size, padded_length].
            input_mask: An int32 0/1 Tensor with shape [batch_size, padded_length].
            target_attrs: An int32 Tensor with shape [batch_size, attr_length].
            reuse_variable: Boolean.
        return:
            mean_loss: A float scalar Tensor
        """

        self.video = video  # [batch_size, length, kernel, kernel, channel]
        self.video_mask = video_mask  # [batch_size, length]
        self.caption = caption  # [batch_size, length]
        self.caption_mask = caption_mask  # [batch_size, length]
        self.target_attribute = target_attribute
        self.train_flag = train_flag

        self.word_embed_t = tf.Variable(self.word_embed, dtype=tf.float32, name="word_embed", trainable=False)
        cells = [self.video_cell, self.attr_cell, self.caption_cell]
        for cell in cells:
            cell = rnn_cell.DropoutWrapper(
                cell,
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob)
            cell = rnn_cell.MultiRNNCell([cell] * self.config.num_layers,
                                         state_is_tuple=False)

        video_emb, video_emb_state = self.build_video_embedding(self.video_cell,
                                                                self.video, self.video_mask, reuse_variable)
        self.attribute, self.concept_loss = self.concept_detector.build_attributes(
            self.attr_cell, self.video, self.video_mask, self.target_attribute, reuse_variable)
        self.caption_loss, predicted_caption = self.build_caption_decoder(
            self.caption_cell, video_emb_state, self.attribute, reuse_variable)

        self.output_words = tf.cast(tf.transpose(tf.stack(predicted_caption), [1, 0]), tf.int32)
        self.mean_loss = self.caption_loss / tf.reduce_sum(self.caption_mask) + self.concept_loss


class CAPTrainer(object):

    def __init__(self, config, model, sess=None, train_summary_dir=None):
        self.sess = sess or tf.get_default_session()
        self.model = model
        self.config = config
        self.train_summary_dir = train_summary_dir

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        learning_rate_decay_fn = None
        if self.config.learning_rate_decay_factor > 0 and self.config.learning_rate_decay_steps > 0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate, global_step,
                    decay_steps=self.config.learning_rate_decay_steps,
                    decay_rate=self.config.learning_rate_decay_factor,
                    staircase=True)
            learning_rate_decay_fn = _learning_rate_decay_fn

        self.no_op = tf.no_op()
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.model.mean_loss,
            global_step=self.global_step,
            learning_rate=self.config.learning_rate,
            learning_rate_decay_fn=learning_rate_decay_fn,
            optimizer=self.config.optimizer,
            clip_gradients=self.config.max_grad_norm,
            summaries=["learning_rate"]
        )
        self.summary_mean_loss = tf.summary.scalar("mean_loss", model.mean_loss)
        self.train_summary_writer = None
        if train_summary_dir is not None:
            self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

    def run_single_step(self, queue, is_train=True):
        start_ts = time.time()

        step_op = self.train_op if is_train else self.no_op
        batch_chunk = queue.get_inputs()
        feed_dict = self.model.get_feed_dict(batch_chunk)
        feed_dict[self.model.train_flag] = is_train
        if not is_train: feed_dict[self.model.dropout_keep_prob] = 1.0

        _, loss, concept_loss, current_step, summary = self.sess.run(
            [step_op, self.model.mean_loss, self.model.concept_loss, self.global_step,
             self.summary_mean_loss],
            feed_dict=feed_dict)
        if self.train_summary_writer is not None:
            self.train_summary_writer.add_summary(summary, current_step)
        end_ts = time.time()
        return loss, concept_loss, current_step, (end_ts - start_ts)

    def eval_single_step(self, val_queue):
        batch_chunk = val_queue.get_inputs()
        feed_dict = self.model.get_feed_dict(batch_chunk)
        feed_dict[self.model.train_flag] = False
        feed_dict[self.model.dropout_keep_prob] = 1.0
        loss, output_words, attribute_indices = self.sess.run([self.model.mean_loss,
                                                               self.model.output_words,
                                                               self.model.attribute],
                                                              feed_dict=feed_dict)
        target_words = batch_chunk['debug_sent']
        return [loss, output_words, target_words, attribute_indices]

    def log_step_message(self, step, loss, concept_loss, step_time, steps_in_epoch, is_train=True):
        log_fn = (is_train and log.info or log.infov)
        batch_size = self.model.batch_size
        log_fn((" [{split_mode:5} step {step:4d} / epoch {epoch:.2f}]  " +
                "batch total-loss: {total_loss:.5f}, concept-loss: {concept_loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) | {train_tag}"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         epoch=float(step)/steps_in_epoch,
                         step=step,
                         total_loss=loss, concept_loss=concept_loss,
                         sec_per_batch=step_time,
                         instance_per_sec=batch_size / step_time,
                         train_tag=self.config.train_tag,
                         )
               )

    def evaluate(self, queue, dataset):
        log.info("Evaluate Phase")
        output_sents = []
        target_sents = []

        iter_length = int(len(dataset) / self.model.batch_size + 1)
        for i in range(iter_length):
            loss, output_words, target_words, attribute_indices = self.eval_single_step(queue)
            batch_output_sents = dataset.assemble_into_sentence(output_words)
            batch_output_sents = [data_util.recover_word(sent) for sent in batch_output_sents]

            output_sents = output_sents + batch_output_sents
            target_sents = target_sents + list(target_words)
            log.infov("{}/{}".format(i, iter_length))
            log.infov("target: {}".format(target_words[0]))
            log.infov("output: {}".format(batch_output_sents[0]))
            attr = []
            for idx in attribute_indices[0]:
                attr.append(dataset.idx2word[idx])
            log.info("attributes: {}".format(attr))
        metrics.compute_score(output_sents, target_sents)
