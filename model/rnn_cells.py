from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf


def mat_weight_mul(mat, weight):
    mat_shape = mat.get_shape().as_list()
    weight_shape = weight.get_shape().as_list()
    assert (mat_shape[-1] == weight_shape[0])
    mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
    mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
    return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])


class GatedAttentionCell(RNNCell):

    def __init__(self, num_units, weights, encoded_question, reuse=None):
        super(GatedAttentionCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.WuQ = weights['WuQ']  # 2H * H
        self.WuP = weights['WuP']  # 2H * H
        self.WvP = weights['WvP']  # H * H
        self.v = weights['v']  # H x 1
        self.Wg = weights['Wg']  # 4H x 4H
        self.uQ = encoded_question
        self._cell = tf.contrib.rnn.GRUCell(num_units)
        self.WuQ_uQ = mat_weight_mul(self.uQ, self.WuQ)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        with vs.variable_scope('attention_pool'):
            utP = inputs  # batch_size * 2H
            vtP = state
            WuQ_uQ = self.WuQ_uQ
            WuP_utP = tf.expand_dims(tf.matmul(utP, self.WuP), 1)  # batch_size x 1 x H
            WvP_vtP = tf.expand_dims(tf.matmul(vtP, self.WvP), 1)  # batch_size x 1 x H

            tanh = tf.tanh(WuQ_uQ + WuP_utP + WvP_vtP)  # batch_size x q_length x h_size

            s_t = mat_weight_mul(tanh, self.v)  # batch_size x q_length x 1
            a_t = tf.nn.softmax(s_t, 1)  # batch_size x q_length x 1
            c_t = tf.reduce_sum(tf.multiply(a_t, self.uQ), 1)  # batch_size x 2h_size

            utP_ct = tf.concat([utP, c_t], 1)  # batch_size x 4H
            g_t = tf.sigmoid(tf.matmul(utP_ct, self.Wg))  # batch_size x 4H
            new_inputs = tf.multiply(g_t, utP_ct)

            return self._cell.call(new_inputs, state)


class GatedAttentionSelfMatchingCell(RNNCell):

    def __init__(self, num_units, weights, encoded_question, reuse=None):
        super(GatedAttentionSelfMatchingCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.WvP = weights['WvP']  # H * H
        self.WvP_hat = weights['WvP_hat']  # H * H
        self.v = weights['v']  # H x 1
        self._cell = tf.contrib.rnn.GRUCell(num_units)

        self.vP = encoded_question
        self.WvP_vP = mat_weight_mul(self.vP, self.WvP)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        with vs.variable_scope('self_matching_pool'):
            vtP = inputs  # batch_size * H
            htP = state
            WvP_vP = self.WvP_vP
            WvP_hat_vtP = tf.expand_dims(tf.matmul(vtP, self.WvP_hat), 1)  # batch_size x 1 x H
            tanh = tf.tanh(WvP_vP + WvP_hat_vtP)  # batch_size x p_length x h_size

            s_t = mat_weight_mul(tanh, self.v)
            a_t = tf.nn.softmax(s_t, 1)
            c_t = tf.reduce_sum(tf.multiply(a_t, self.vP), 1)
            vtP_ct = tf.concat([vtP, c_t], 1)  # batch_size x 2H

            return self._cell.call(vtP_ct, state)


class PointerGRUCell(RNNCell):

    def __init__(self, num_units, weights, encoded_question, reuse=None):
        super(PointerGRUCell, self).__init__(_reuse=reuse)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        pass
