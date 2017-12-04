# TODO: add bias
import math
import tensorflow as tf
from rnn_cells import mat_weight_mul, GatedAttentionCell, GatedAttentionSelfMatchingCell, PointerGRUCell

class RNet:

    @staticmethod
    def random_weight(dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def __init__(self, options):
        self.options = options
        h = options['h_size']
        with tf.device('/cpu:0'):
            self.WuQ = self.random_weight(2 * h, h, name='WuQ')
            self.WuP = self.random_weight(2 * h, h, name='WuP')
            self.WvP = self.random_weight(h, h, name='WvP')
            self.v = self.random_weight(h, 1, name='v')
            self.Wg = self.random_weight(4 * h, 4 * h, name='Wg')
            self.WvP_hat = self.random_weight(h, h, name='WvP_hat')
            self.WvQ = self.random_weight(1, h, name='WvQ')

    def build_model(self, input_pipeline):
        options = self.options
        # placeholders
        batch_size = options['batch_size']
        p_length = options['p_length']
        q_length = options['q_length']
        emb_dim = options['emb_dim']

        eP = input_pipeline['eP']
        eQ = input_pipeline['eQ']
        asi = input_pipeline['asi']
        aei = input_pipeline['aei']
        # eP = tf.placeholder(tf.float32, [batch_size, p_length, emb_dim])
        # eQ = tf.placeholder(tf.float32, [batch_size, q_length, emb_dim])
        # asi = tf.placeholder(tf.float32, [batch_size, p_length])
        # aei = tf.placeholder(tf.float32, [batch_size, p_length])

        print('Shape of eP: {}'.format(eP.get_shape()))
        print('Shape of eQ: {}'.format(eQ.get_shape()))
        print('Shape of asi: {}'.format(asi.get_shape()))
        print('Shape of aei: {}'.format(aei.get_shape()))

        # embeddings concatenation
        # TODO: add character level embeddings
        eQcQ = eQ
        ePcP = eP

        # Question and Passage Encoding
        # TODO: test dynamic bi-rnn, w/o stack

        ## difference: # of GRU layers, dropout application
        h_size = options['h_size']
        in_keep_prob = options['in_keep_prob']
        with tf.variable_scope('encoding') as scope:
            # TODO: number of layers as parameter
            gru_cells_fw = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(h_size)
                                                        for _ in range(1)])
            gru_cells_bw = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(h_size)
                                                        for _ in range(1)])

            uQ_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw, gru_cells_bw, eQcQ, dtype=tf.float32, scope='context_encoding')
            tf.get_variable_scope().reuse_variables()

            uP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw, gru_cells_bw, ePcP, dtype=tf.float32, scope='context_encoding')
            uQ = tf.concat(uQ_2, 2)
            uP = tf.concat(uP_2, 2)
            uQ = tf.nn.dropout(uQ, in_keep_prob)
            uP = tf.nn.dropout(uP, in_keep_prob)
            print('Shape of uP: {}'.format(uP.get_shape()))
            print('Shape of uQ: {}'.format(uQ.get_shape()))

        # Question and passage matching
        # Note: it is not clear here if bi-rnn or rnn should be used
        with tf.variable_scope('attention_matching'):
            weights = {
                'WuQ': self.WuQ,
                'WuP': self.WuP,
                'WvP': self.WvP,
                'v': self.v,
                'Wg': self.Wg
            }

            # attn_cells_fw = tf.nn.rnn_cell.MultiRNNCell([GatedAttentionCell(h_size, weights, uQ)
            #                                             for _ in range(1)])
            # attn_cells_bw = tf.nn.rnn_cell.MultiRNNCell([GatedAttentionCell(h_size, weights, uQ)
            #                                             for _ in range(1)])
            # vP_2, _ = tf.nn.bidirectional_dynamic_rnn(attn_cells_fw,
            #                                        attn_cells_bw,
            #                                        inputs=uP,
            #                                        dtype=tf.float32)
            # vP = tf.concat(vP_2, 2)
            attn_cells = tf.nn.rnn_cell.MultiRNNCell([GatedAttentionCell(h_size, weights, uQ)
                                                      for _ in range(1)])
            vP, _ = tf.nn.dynamic_rnn(cell=attn_cells, dtype=tf.float32, inputs=uP)

            print('Shape of vP: {}'.format(vP.get_shape()))

        # self matching layer
        with tf.variable_scope('self_matching'):
            weights = {
                'WvP': self.WvP,
                'v': self.v,
                'WvP_hat': self.WvP_hat
            }
            attn_sm_cells_fw = tf.nn.rnn_cell.MultiRNNCell([GatedAttentionSelfMatchingCell(h_size, weights, vP)
                                                            for _ in range(1)])
            attn_sm_cells_bw = tf.nn.rnn_cell.MultiRNNCell([GatedAttentionSelfMatchingCell(h_size, weights, vP)
                                                            for _ in range(1)])

            hP_2, _ = tf.nn.bidirectional_dynamic_rnn(attn_sm_cells_fw,
                                                      attn_sm_cells_bw,
                                                      inputs=vP,
                                                      dtype=tf.float32)
            hP = tf.concat(hP_2, 2)
            print('Shape of hP: {}'.format(hP.get_shape()))

        # output layer
        with tf.variable_scope('output_layer'):
            # quote in section 4.2: After the original self-matching layer of the passage,
            # we utilize bi-directional GRU to deeply integrate the matching
            # results before feeding them into answer pointer layer.
            gru_cells_fw2 = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(h_size)
                                                         for _ in range(1)])
            gru_cells_bw2 = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(h_size)
                                                         for _ in range(1)])

            gP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw2, gru_cells_bw2, hP, dtype=tf.float32, scope='deeply_integration')
            gP = tf.concat(gP_2, 2)
            gP = tf.nn.dropout(gP, in_keep_prob)

            # question pooling
            WuQ_uQ = mat_weight_mul(uQ, self.WuQ)  # batch_size x q_length x H
            tanh = tf.tanh(WuQ_uQ + self.WvQ)
            print('Shape of WuQ_uQ: {}'.format(WuQ_uQ.get_shape()))

            s = mat_weight_mul(tanh, self.v)
            a = tf.nn.softmax(s, 1)
            rQ = tf.reduce_sum(tf.multiply(a, uQ), 1)
            print('Shape of rQ: {}'.format(rQ.get_shape()))