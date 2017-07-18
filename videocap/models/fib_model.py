from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import time

from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.layers import batch_norm

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


class FIBGenerator(object):
    """
    """
    def __init__(self, config, word_embed, attr_idx):
        self.config = config
        self.batch_size = config.batch_size
        self.word_embed = word_embed
        self.vocab_size = word_embed.shape[0]
        self.name = 'Fill-In-the-BlankGenerator'
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
        self.fw_cell = _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=False)
        self.bw_cell = _lstm_map[self.config.lstm_cell](
            num_units=self.config.hidden_dim, state_is_tuple=False)

        self.concept_detector = \
            _concept_detector_map[self.config.concept_detector](config=config, attr_idx=attr_idx)
        self.attention_method = \
            _attention_method_map[self.config.attention_method](config=config)

    def get_feed_dict(self, batch_chunk):
        feed_dict = {
            self.video: batch_chunk['video_features'].astype(float),
            self.video_mask: batch_chunk['video_mask'].astype(float),
            self.blank_caption: batch_chunk['blank_sent'].astype(float),
            self.blank_caption_mask: batch_chunk['blank_sent_mask'].astype(float),
            self.answer: batch_chunk['answer'],
            self.target_attribute: batch_chunk['bow'].astype(float)
        }
        return feed_dict

    def get_placeholder(self):
        video = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps, 7, 7, 2048])
        video_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.video_steps])
        blank_caption = tf.placeholder(tf.int32, [self.config.batch_size, self.config.caption_length])
        blank_caption_mask = tf.placeholder(tf.float32, [self.config.batch_size, self.config.caption_length])
        target_attribute = tf.placeholder(tf.float32, [self.config.batch_size, len(self.attr_idx)])
        answer = tf.placeholder(tf.float32, [self.config.batch_size, self.vocab_size])
        train_flag = tf.placeholder(tf.bool)

        result = {
            'video': video,
            'video_mask': video_mask,
            'blank_caption': blank_caption,
            'blank_caption_mask': blank_caption_mask,
            'answer': answer,
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

    def build_fib_decoder(self,
                          fw_cell,
                          bw_cell,
                          vid_emb_state,
                          attributes,
                          reuse_variable=False):

        embed_attributes = self.build_caption_embedding(attributes, name="embedding_attributes")
        embedded_sentence = self.build_caption_embedding(self.blank_caption, name="embedding_blank_sent")

        with tf.variable_scope("attended_input", reuse=reuse_variable) as scope:
            self.sentence_list = []
            self.gammas = []

            for i in range(self.config.caption_length):
                if i > 0:
                    scope.reuse_variables()

                attended_sentence, gamma = self.attention_method.attended_input(
                    embedded_sentence[:, i, :], embed_attributes)
                self.sentence_list.append(attended_sentence)
                self.gammas.append(gamma)
            self.gammas = tf.stack(self.gammas)
            self.gammas = tf.transpose(self.gammas, [1, 0, 2])

        with tf.variable_scope("fib_rnn", reuse=reuse_variable) as scope:
            outputs, _, _ = static_bidirectional_rnn(self.fw_cell,
                                                     self.bw_cell,
                                                     self.sentence_list,
                                                     initial_state_fw=vid_emb_state,
                                                     initial_state_bw=vid_emb_state)
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            masked_outputs = outputs * tf.expand_dims(self.blank_caption_mask, 2)
            rnn_output = tf.reduce_sum(masked_outputs, 1)

        with tf.variable_scope("affine", reuse=reuse_variable) as scope:
            affine_output = slim.fully_connected(inputs=rnn_output,
                                                 num_outputs=self.config.hidden_dim,
                                                 activation_fn=tf.tanh,
                                                 weights_initializer=self.initializer,
                                                 scope=scope,
                                                 reuse=reuse_variable)
            affine_dropout = tf.nn.dropout(affine_output, self.dropout_keep_prob)

        with tf.variable_scope("attended_output", reuse=reuse_variable) as scope:
            attended_output, beta = self.attention_method.attended_output(
                affine_dropout, embed_attributes)
            attended_output_dropout = tf.nn.dropout(attended_output,
                                                    self.dropout_keep_prob)

        attended_output_BN = batch_norm(attended_output_dropout, is_training=self.train_flag, scale=True)

        with tf.variable_scope("prediction", reuse=reuse_variable):
            scores = slim.fully_connected(inputs=attended_output_BN,
                                          num_outputs=self.vocab_size,
                                          activation_fn=None,
                                          weights_initializer=self.initializer,
                                          scope=scope,
                                          reuse=reuse_variable)
            predictions = tf.argmax(scores, 1)

        with tf.variable_scope("acc_and_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.answer, name="word_loss")
            losses = tf.reduce_mean(losses)
            self.correct_predictions = tf.equal(predictions, tf.argmax(self.answer, 1))
            acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        return losses, predictions, acc

    def build_model(self,
                    video,
                    video_mask,
                    blank_caption,
                    blank_caption_mask,
                    target_attribute,
                    answer,
                    train_flag,
                    reuse_variable=False):

        self.video = video
        self.video_mask = video_mask
        self.blank_caption = blank_caption
        self.blank_caption_mask = blank_caption_mask
        self.answer = answer
        self.target_attribute = target_attribute
        self.train_flag = train_flag

        self.word_embed_t = tf.Variable(self.word_embed, dtype=tf.float32, name="word_embed", trainable=False)
        cells = [self.video_cell, self.attr_cell, self.fw_cell, self.bw_cell]
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

        self.fib_loss, self.predictions, self.acc = self.build_fib_decoder(
            self.fw_cell, self.bw_cell, video_emb_state, self.attribute, reuse_variable)

        self.mean_loss = self.fib_loss + self.concept_loss


class FIBTrainer(object):

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

        self.build_eval_graph()

    def build_eval_graph(self):
        self.total_correct = tf.Variable(0.0, trainable=False, collections=[])
        self.example_count = tf.Variable(0.0, trainable=False, collections=[])

        self.accuracy = self.total_correct / self.example_count
        inc_total_correct = self.total_correct.assign_add(
            tf.reduce_sum(tf.cast(self.model.correct_predictions, "float")))
        inc_example_count = self.example_count.assign_add(self.model.batch_size)

        with tf.control_dependencies([self.total_correct.initializer,
                                      self.example_count.initializer]):
            self.eval_reset = tf.no_op()

        with tf.control_dependencies([inc_total_correct, inc_example_count]):
            self.eval_step = tf.no_op()

    def run_single_step(self, queue, is_train=True):
        start_ts = time.time()

        step_op = self.train_op if is_train else self.no_op
        batch_chunk = queue.get_inputs()
        feed_dict = self.model.get_feed_dict(batch_chunk)
        feed_dict[self.model.train_flag] = is_train
        if not is_train: feed_dict[self.model.dropout_keep_prob] = 1.0

        _, loss, acc, concept_loss, current_step, summary = self.sess.run(
            [step_op, self.model.mean_loss, self.model.acc, self.model.concept_loss, self.global_step,
             self.summary_mean_loss],
            feed_dict=feed_dict)
        if self.train_summary_writer is not None:
            self.train_summary_writer.add_summary(summary, current_step)
        end_ts = time.time()
        result = {
            "loss": loss,
            "concept_loss": concept_loss,
            "acc": acc,
            "current_step": current_step,
            "step_time": (end_ts - start_ts)
        }
        return result

    def eval_single_step(self, val_queue):
        batch_chunk = val_queue.get_inputs()
        feed_dict = self.model.get_feed_dict(batch_chunk)
        feed_dict[self.model.train_flag] = False
        feed_dict[self.model.dropout_keep_prob] = 1.0
        _, loss, predictions, attribute_indices = self.sess.run([self.eval_step,
                                                                 self.model.mean_loss,
                                                                 self.model.predictions,
                                                                 self.model.attribute],
                                                                feed_dict=feed_dict)
        target_indices = np.argmax(batch_chunk['answer'], axis=1)
        return [loss, predictions, target_indices, attribute_indices]

    def log_step_message(self, current_step, loss, acc, concept_loss, step_time, steps_in_epoch, is_train=True):
        log_fn = (is_train and log.info or log.infov)
        batch_size = self.model.batch_size
        log_fn((" [{split_mode:5} step {step:4d} / epoch {epoch:.2f}]  " +
                "batch total-loss: {total_loss:.5f}, concept-loss: {concept_loss:.5f}, accuracy: {acc:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) | {train_tag}"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         epoch=float(current_step)/steps_in_epoch,
                         step=current_step,
                         total_loss=loss, concept_loss=concept_loss,
                         acc=acc,
                         sec_per_batch=step_time,
                         instance_per_sec=batch_size / step_time,
                         train_tag=self.config.train_tag,
                         )
               )

    def evaluate(self, queue, dataset):
        log.info("Evaluate Phase")
        iter_length = int(len(dataset) / self.model.batch_size + 1)
        self.sess.run(self.eval_reset)

        for i in range(iter_length):
            loss, predictions, target_indices, attribute_indices = self.eval_single_step(queue)
            target_word = dataset.idx2word[target_indices[0]]
            output_word = dataset.idx2word[predictions[0]]
            log.infov("[FIB {step:3d}/{total_length:3d}] target: {target}, prediction: {prediction}".format(
                step=i, total_length=iter_length, target=target_word, prediction=output_word))
            attr = []
            for idx in attribute_indices[0]:
                attr.append(dataset.idx2word[idx])
            log.info("concepts: {}".format(attr))
        total_acc = self.sess.run(self.accuracy)
        log.infov("[FIB] total accurycy: {acc:.5f}".format(total_acc))
