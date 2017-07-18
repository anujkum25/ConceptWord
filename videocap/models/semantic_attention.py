import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class SemanticAttention(object):

    def __init__(self, config=None):
        self.config = config

    def _bilinear_operation(self, y, W, b, A):
        _, dim = y.get_shape().as_list()
        batch_size, attr_length, word_dim = A.get_shape().as_list()

        y_W = tf.matmul(y, W)
        y_W = tf.reshape(y_W, [batch_size, -1, word_dim])
        y_W_A = tf.matmul(y_W, A, adjoint_b=True) + b

        return y_W_A

    def _attended_sum(self, att_ratio, seqs):
        """
        att_ratio: [batch_size, length]
        seqs: [batch_size, length, dim]
        """
        exp_ratio = tf.expand_dims(att_ratio, 2)
        attended_sum = tf.reduce_sum(exp_ratio*seqs, 1)
        # [batch_size, dim]

        return attended_sum

    def _get_attention(self, input, embed_attributes, U, b):
        batch_size, dim = input.get_shape().as_list()
        _, attr_length, word_dim = embed_attributes.get_shape().as_list()

        reshaped_input = tf.reshape(input, [-1, dim])
        alpha = self._bilinear_operation(reshaped_input, U, b, embed_attributes)
        alpha = tf.squeeze(alpha)
        # [batch_size, attr_length]

        return tf.nn.softmax(alpha)

    def attended_output(self, outputs, embed_attributes):
        batch_size, dim = outputs.get_shape().as_list()
        _, length, word_dim = embed_attributes.get_shape().as_list()

        W = tf.get_variable("output_W",
                            [dim, word_dim],
                            initializer=xavier_initializer(uniform=False))
        b = tf.get_variable("output_b",
                            [], initializer=tf.constant_initializer(0))
        w = tf.get_variable("output_w", [1, dim],
                            initializer=xavier_initializer(uniform=False))

        tanh_attributes = tf.tanh(embed_attributes)
        beta = self._get_attention(outputs, tanh_attributes, W, b)
        tr_attributes = tf.matmul(tf.reshape(tanh_attributes, [-1, word_dim]),
                                  W, transpose_b=True)
        tr_attributes.get_shape().assert_is_compatible_with([batch_size*length, dim])
        tr_attributes = tf.reshape(tr_attributes, [batch_size, length, dim])
        attended_attributes = self._attended_sum(beta, tr_attributes)
        output_p = outputs + w * attended_attributes

        output_p = tf.reshape(output_p, [-1, dim])

        return output_p, beta

    def attended_input(self, embed_word, embed_attributes):
        batch_size, word_dim = embed_word.get_shape().as_list()

        W = tf.get_variable("input_W",
                            [word_dim, word_dim],
                            initializer=xavier_initializer(uniform=False))
        b = tf.get_variable("input_b",
                            [], initializer=tf.constant_initializer(0))
        w = tf.get_variable("input_w", [1, word_dim],
                            initializer=xavier_initializer(uniform=False))

        gamma = self._get_attention(embed_word, embed_attributes, W, b)
        attended_attributes = self._attended_sum(gamma, embed_attributes)
        input_x = embed_word + w * attended_attributes

        return input_x, gamma
