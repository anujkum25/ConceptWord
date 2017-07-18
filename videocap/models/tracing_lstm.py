import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer_conv2d


class TracingLSTM(object):

    def __init__(self, config=None, attr_idx=None):
        self.config = config
        self.attr_idx = tf.constant(attr_idx)
        self.attr_length = config.attr_length

    def _operate_attr_attention(self, att_map, video_frame):
        batch_size, width, height, local_size = att_map.get_shape().as_list()

        reshaped_att_map = tf.reshape(att_map, [batch_size, width*height, local_size])
        transposed_att_map = tf.transpose(reshaped_att_map, [0, 2, 1])

        reshaped_frm = tf.reshape(video_frame, [batch_size, width*height, -1])
        attentioned_frm = tf.matmul(transposed_att_map, reshaped_frm)
        attentioned_frm = tf.reshape(attentioned_frm, [batch_size*local_size, -1])

        return attentioned_frm

    def _conv_forward_prop(self, input, kernel_shape, strides, name):
        with tf.variable_scope(name) as scope:
            history = tf.get_collection('__attr_conv')
            if scope.name in history:
                scope.reuse_variables()
            tf.add_to_collection('__attr_conv', scope.name)
            W = tf.get_variable("kernel",
                                shape=kernel_shape,
                                initializer=xavier_initializer_conv2d(uniform=False))
            b = tf.get_variable("bias",
                                shape=[kernel_shape[-1]],
                                initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(input, W, strides=strides, padding='SAME')
            return tf.nn.bias_add(conv, b)

    def _get_att_map(self, output, next_input, mask, init_att_map, scope):
        # output : [batch_size*local_size, hidden_dim]
        # next_input : next time video frame feature
        batch_size, width, height, video_dim = next_input.get_shape().as_list()
        local_size = width * height
        hidden_dim = self.config.hidden_dim

        reshape_init_att_map = tf.transpose(init_att_map, [0, 3, 1, 2])
        # [batch_size, local_size, kernel_size, kernel_size]
        reshape_init_att_map = tf.reshape(reshape_init_att_map,
                                          [batch_size*local_size,
                                           width, height])
        # reduce dimension of conv feature
        next_input = tf.tanh(next_input)

        reshaped_output = tf.reshape(output, [batch_size, local_size, hidden_dim])
        expand_output = tf.expand_dims(reshaped_output, axis=2)
        # [batch_size, local_size, 1, hidden_dim]

        reshaped_input = tf.reshape(next_input, [batch_size, width*height, hidden_dim])
        expand_input = tf.expand_dims(reshaped_input, axis=1)
        # [batch_size, 1, kernel_size*kernel_size, hidden_dim]

        mult_ret = tf.multiply(expand_input, expand_output)
        normalized_ret = tf.nn.l2_normalize(mult_ret, dim=2)
        normalized_ret.get_shape().assert_is_compatible_with([batch_size, local_size, width*height, hidden_dim])
        normalized_ret = tf.reshape(normalized_ret, [batch_size*local_size, width, height, hidden_dim])

        conv1 = self._conv_forward_prop(normalized_ret,
                                        [3, 3, hidden_dim, hidden_dim],
                                        [1, 1, 1, 1],
                                        name="conv1")
        relu1 = tf.nn.relu(conv1)
        conv2 = self._conv_forward_prop(relu1,
                                        [3, 3, hidden_dim, 1],
                                        [1, 1, 1, 1],
                                        name="conv2")
        # conv2 : [batch_size*local_size, kernel_size, kernel_size, 1]
        attention_map = tf.reshape(conv2, [batch_size*local_size, local_size])
        attention_map = tf.nn.softmax(attention_map)

        att_maps = tf.reshape(attention_map, [batch_size*local_size, width, height])
        mask_expand = tf.expand_dims(tf.expand_dims(mask, -1), -1)
        # mask_expand [batch_size*local_size, 1, 1] (video_mask)
        # if mask value is 0, start init attention else apply att_maps
        att_maps = tf.multiply(mask_expand, att_maps) + tf.multiply(1-mask_expand, reshape_init_att_map)
        att_maps = tf.reshape(att_maps, [batch_size, local_size, width, height])
        # TODO check!
        att_maps = tf.transpose(att_maps, [0, 2, 3, 1])
        att_maps.get_shape().assert_is_compatible_with([batch_size, width, height, local_size])
        return att_maps

    def _video_embedding(self, videos, video_input_dim):
        batch_size, time_steps, width, height, video_dim = videos.get_shape().as_list()
        W = tf.get_variable("kernel",
                            [3, 3, video_dim, video_input_dim],
                            initializer=xavier_initializer_conv2d(uniform=False))
        pooled = tf.nn.max_pool(tf.reshape(videos, [-1, width, height, video_dim]),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
        pooled_kernel_size = pooled.get_shape().as_list()[1]
        video_embedding = tf.nn.conv2d(pooled,
                                       W, strides=[1, 1, 1, 1], padding='SAME')
        video_embedding = tf.reshape(video_embedding,
                                     [batch_size, time_steps, pooled_kernel_size, pooled_kernel_size, video_input_dim])
        return video_embedding

    def build_attributes(self, attr_cell, videos, video_masks, target_attrs,
                         reuse_variable=False):
        with tf.variable_scope("attr_video_embedding", reuse=reuse_variable):
            videos = self._video_embedding(videos, attr_cell.output_size)

        batch_size, time_steps, width, height, video_dim = videos.get_shape().as_list()
        _, vocab_size = target_attrs.get_shape().as_list()
        local_size = width * height

        expand_video_mask = tf.tile(tf.expand_dims(video_masks, 1), [1, local_size, 1])
        expand_video_mask = tf.reshape(expand_video_mask,
                                       [batch_size*local_size, time_steps])
        init_att_map = tf.constant(np.eye(local_size).reshape(
            [width, height, local_size]), dtype=tf.float32)
        init_att_map = tf.tile(tf.expand_dims(init_att_map, 0), [batch_size, 1, 1, 1])
        initial_state = tf.zeros([batch_size*local_size, attr_cell.state_size])
        attr_rnn_states = [initial_state]
        attr_rnn_outputs = []

        att_maps = [init_att_map]
        with tf.variable_scope("attr_rnn", reuse=reuse_variable) as scope:
            for i in range(time_steps):
                if i > 0:
                    scope.reuse_variables()
                attentioned_input = self._operate_attr_attention(att_maps[-1],
                                                                 videos[:, i, :, :, :])
                new_output, new_state = attr_cell(attentioned_input, attr_rnn_states[-1])
                attr_rnn_outputs.append(new_output)
                attr_rnn_states.append(new_state)
                if i < (time_steps - 1):
                    att_maps.append(self._get_att_map(new_output,
                                                      videos[:, i+1, :, :, :],
                                                      expand_video_mask[:, i],
                                                      init_att_map,
                                                      scope))

        outputs = tf.reshape(attr_rnn_outputs[-1], [batch_size, width, height, self.config.hidden_dim])
        with tf.variable_scope("attr_logit", reuse=reuse_variable):
            K = tf.get_variable("K",
                                shape=[width, height, self.config.hidden_dim, vocab_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            att_affine = tf.nn.conv2d(outputs,
                                      K,
                                      strides=[1, 1, 1, 1],
                                      padding='VALID')
            logits = tf.squeeze(att_affine)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=target_attrs)
            loss = tf.reduce_mean(loss)

        with tf.name_scope("top_attr"):
            with tf.device("/cpu:0"):
                _, top_indices = tf.nn.top_k(logits, self.attr_length)
                attributes = tf.nn.embedding_lookup(self.attr_idx, top_indices)
        return attributes, loss
