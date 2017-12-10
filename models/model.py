# TODO: add bias
import math
import tensorflow as tf
from .rnn_cells import mat_weight_mul, GatedAttentionCell, GatedAttentionSelfMatchingCell, PointerGRUCell


class RNet:
    @staticmethod
    def random_weight(dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    @staticmethod
    def dropout_wrapped_grucell(hidden_size, in_keep_prob, name=None):
        cell = tf.contrib.rnn.GRUCell(hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def __init__(self, options):
        self.options = options
        h = options['h_size']
        with tf.device('/cpu:0'):
            self.WuQ = self.random_weight(2 * h, h, name='WuQ')
            self.WuP = self.random_weight(2 * h, h, name='WuP')
            self.WvP = self.random_weight(h, h, name='WvP')
            self.v = self.random_weight(h, 1, name='v')
            self.Wg = self.random_weight(4 * h, 4 * h, name='Wg')
            self.Wg2 = self.random_weight(2 * h, 2 * h, name='Wg')
            self.WvP_hat = self.random_weight(h, h, name='WvP_hat')
            self.WvQ = self.random_weight(1, h, name='WvQ')
            self.Wha = self.random_weight(2 * h, h, name='Wha')
            self.WhP = self.random_weight(2 * h, h, name='WhP')

    def build_model(self, it):
        options = self.options
        # placeholders
        batch_size = options['batch_size']
        p_length = options['p_length']
        q_length = options['q_length']
        emb_dim = options['emb_dim']

        eP = it['eP']
        eQ = it['eQ']
        asi = it['asi']
        aei = it['aei']

        print('Shape of eP: {}'.format(eP.get_shape()))
        print('Shape of eQ: {}'.format(eQ.get_shape()))
        print('Shape of asi: {}'.format(asi.get_shape()))
        print('Shape of aei: {}'.format(aei.get_shape()))

        # embeddings concatenation
        # TODO: add character level embeddings
        eQcQ = eQ
        ePcP = eP

        ## difference: # of GRU layers, dropout application
        h_size = options['h_size']
        in_keep_prob = options['in_keep_prob']
        with tf.variable_scope('encoding') as scope:
            # TODO: number of layers as parameter
            gru_cells_fw = tf.nn.rnn_cell.MultiRNNCell([self.dropout_wrapped_grucell(h_size, in_keep_prob)
                                                        for _ in range(3)])
            gru_cells_bw = tf.nn.rnn_cell.MultiRNNCell([self.dropout_wrapped_grucell(h_size, in_keep_prob)
                                                        for _ in range(3)])

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

            attn_cells = GatedAttentionCell(h_size, weights, uQ)
            vP, _ = tf.nn.dynamic_rnn(cell=attn_cells, dtype=tf.float32, inputs=uP)
            vP = tf.nn.dropout(vP, in_keep_prob)

            print('Shape of vP: {}'.format(vP.get_shape()))

        # self matching layer
        with tf.variable_scope('self_matching'):
            weights = {
                'WvP': self.WvP,
                'v': self.v,
                'WvP_hat': self.WvP_hat,
                'Wg2': self.Wg2
            }
            attn_sm_cells_fw = GatedAttentionSelfMatchingCell(h_size, weights, vP)
            attn_sm_cells_bw = GatedAttentionSelfMatchingCell(h_size, weights, vP)
            hP_2, _ = tf.nn.bidirectional_dynamic_rnn(attn_sm_cells_fw,
                                                      attn_sm_cells_bw,
                                                      inputs=vP,
                                                      dtype=tf.float32)
            hP = tf.concat(hP_2, 2)
            hP = tf.nn.dropout(hP, in_keep_prob)
            print('Shape of hP: {}'.format(hP.get_shape()))

        # output layer
        with tf.variable_scope('output_layer'):
            # quote in section 4.2: After the original self-matching layer of the passage,
            # we utilize bi-directional GRU to deeply integrate the matching
            # results before feeding them into answer pointer layer.
            gru_cells_fw2 = tf.contrib.rnn.GRUCell(h_size)
            gru_cells_bw2 = tf.contrib.rnn.GRUCell(h_size)
            gP_2, _ = tf.nn.bidirectional_dynamic_rnn(
                gru_cells_fw2, gru_cells_bw2, hP, dtype=tf.float32, scope='deeply_integration')
            gP = tf.concat(gP_2, 2)

            # question pooling
            WuQ_uQ = mat_weight_mul(uQ, self.WuQ)  # batch_size x q_length x H
            tanh = tf.tanh(WuQ_uQ + self.WvQ)
            s = mat_weight_mul(tanh, self.v)
            a = tf.nn.softmax(s, 1)
            rQ = tf.reduce_sum(tf.multiply(a, uQ), 1)
            rQ = tf.nn.dropout(rQ, in_keep_prob)
            print('Shape of rQ: {}'.format(rQ.get_shape()))

            # PointerNet
            s = []
            pt = []
            Whp_hP = mat_weight_mul(gP, self.WhP)
            htm1a = rQ
            output_cell = tf.contrib.rnn.GRUCell(2 * h_size)
            for i in range(2):
                Wha_htm1a = tf.expand_dims(tf.matmul(htm1a, self.Wha), 1)
                tanh = tf.tanh(Whp_hP + Wha_htm1a)
                st = mat_weight_mul(tanh, self.v)
                s.append(tf.squeeze(st))
                at = tf.nn.softmax(st, 1)
                pt.append(tf.argmax(at, 1))
                ct = tf.reduce_sum(tf.multiply(at, gP), 1)
                _, htm1a = output_cell.call(ct, htm1a)

            p = tf.concat(pt, 1)
            print(p)


        with tf.variable_scope('loss_accuracy'):
            as_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(asi),
                                                                     logits=s[0])
            ae_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(aei),
                                                                     logits=s[1])
            loss = (as_loss + ae_loss) / 2.0

            as_accu, _ = tf.metrics.accuracy(labels=tf.squeeze(asi), predictions=pt[0])
            ae_accu, _ = tf.metrics.accuracy(labels=tf.squeeze(aei), predictions=pt[1])

            accu = (ae_accu + as_accu) / 2.0


        return loss, p, accu


class RNet2:
    def random_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def random_bias(self, dim, name=None):
        return tf.Variable(tf.truncated_normal([dim]), name=name)

    def random_scalar(self, name=None):
        return tf.Variable(0.0, name=name)

    def DropoutWrappedGRUCell(self, hidden_size, in_keep_prob, name=None):
        # cell = tf.contrib.rnn.GRUCell(hidden_size)
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def __init__(self, options):
        with tf.device('/cpu:0'):
            self.options = options

            # Char embeddings
            if options['char_emb']:
                self.char_emb_mat = self.random_weight(self.options['char_vocab_size'],
                                                       self.options['char_emb_mat_dim'], name='char_emb_matrix')

            # Weights
            self.W_uQ = self.random_weight(2 * options['state_size'], options['state_size'], name='W_uQ')
            self.W_uP = self.random_weight(2 * options['state_size'], options['state_size'], name='W_uP')
            self.W_vP = self.random_weight(options['state_size'], options['state_size'], name='W_vP')
            self.W_g_QP = self.random_weight(4 * options['state_size'], 4 * options['state_size'], name='W_g_QP')
            self.W_smP1 = self.random_weight(options['state_size'], options['state_size'], name='W_smP1')
            self.W_smP2 = self.random_weight(options['state_size'], options['state_size'], name='W_smP2')
            self.W_g_SM = self.random_weight(2 * options['state_size'], 2 * options['state_size'], name='W_g_SM')
            self.W_ruQ = self.random_weight(2 * options['state_size'], 2 * options['state_size'], name='W_ruQ')
            self.W_vQ = self.random_weight(options['state_size'], 2 * options['state_size'], name='W_vQ')
            self.W_VrQ = self.random_weight(options['q_length'], options['state_size'],
                                            name='W_VrQ')  # has same size as u_Q
            self.W_hP = self.random_weight(2 * options['state_size'], options['state_size'], name='W_hP')
            self.W_ha = self.random_weight(2 * options['state_size'], options['state_size'], name='W_ha')

            # Biases
            self.B_v_QP = self.random_bias(options['state_size'], name='B_v_QP')
            self.B_v_SM = self.random_bias(options['state_size'], name='B_v_SM')
            self.B_v_rQ = self.random_bias(2 * options['state_size'], name='B_v_rQ')
            self.B_v_ap = self.random_bias(options['state_size'], name='B_v_ap')

            # QP_match
            with tf.variable_scope('QP_match') as scope:
                self.QPmatch_cell = self.DropoutWrappedGRUCell(self.options['state_size'], self.options['in_keep_prob'])
                self.QPmatch_state = self.QPmatch_cell.zero_state(self.options['batch_size'], dtype=tf.float32)

            # Ans Ptr
            with tf.variable_scope('Ans_ptr') as scope:
                self.AnsPtr_cell = self.DropoutWrappedGRUCell(2 * self.options['state_size'],
                                                              self.options['in_keep_prob'])

    def build_model(self, it):
        opts = self.options

        # placeholders
        paragraph = it['eP']
        question = it['eQ']
        answer_si = it['asi']
        answer_ei = it['aei']
        eQcQ = question
        ePcP = paragraph

        unstacked_eQcQ = tf.unstack(eQcQ, opts['q_length'], 1)
        unstacked_ePcP = tf.unstack(ePcP, opts['p_length'], 1)
        with tf.variable_scope('encoding') as scope:
            stacked_enc_fw_cells = [self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob']) for _ in
                                    range(2)]
            stacked_enc_bw_cells = [self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob']) for _ in
                                    range(2)]
            q_enc_outputs, q_enc_final_fw, q_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
                stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_eQcQ, dtype=tf.float32, scope='context_encoding')
            tf.get_variable_scope().reuse_variables()
            p_enc_outputs, p_enc_final_fw, p_enc_final_bw = tf.contrib.rnn.stack_bidirectional_rnn(
                stacked_enc_fw_cells, stacked_enc_bw_cells, unstacked_ePcP, dtype=tf.float32, scope='context_encoding')
            u_Q = tf.stack(q_enc_outputs, 1)
            u_P = tf.stack(p_enc_outputs, 1)
        u_Q = tf.nn.dropout(u_Q, opts['in_keep_prob'])
        u_P = tf.nn.dropout(u_P, opts['in_keep_prob'])
        print(u_Q)
        print(u_P)

        v_P = []
        print('Question-Passage Matching')
        for t in range(opts['p_length']):
            # Calculate c_t
            W_uQ_u_Q = self.mat_weight_mul(u_Q, self.W_uQ)  # [batch_size, q_length, state_size]
            tiled_u_tP = tf.concat([tf.reshape(u_P[:, t, :], [opts['batch_size'], 1, -1])] * opts['q_length'], 1)
            W_uP_u_tP = self.mat_weight_mul(tiled_u_tP, self.W_uP)

            if t == 0:
                tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)
            else:
                tiled_v_t1P = tf.concat([tf.reshape(v_P[t - 1], [opts['batch_size'], 1, -1])] * opts['q_length'], 1)
                W_vP_v_t1P = self.mat_weight_mul(tiled_v_t1P, self.W_vP)
                tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P)
            s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_QP, [-1, 1])))
            a_t = tf.nn.softmax(s_t, 1)
            tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'], 2)
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), 1)  # [batch_size, 2 * state_size]

            # gate
            u_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(u_P[:, t, :]), c_t], 1), 1)
            g_t = tf.sigmoid(self.mat_weight_mul(u_tP_c_t, self.W_g_QP))
            u_tP_c_t_star = tf.squeeze(tf.multiply(u_tP_c_t, g_t))

            # QP_match
            with tf.variable_scope("QP_match"):
                if t > 0: tf.get_variable_scope().reuse_variables()
                output, self.QPmatch_state = self.QPmatch_cell(u_tP_c_t_star, self.QPmatch_state)
                v_P.append(output)
        v_P = tf.stack(v_P, 1)
        v_P = tf.nn.dropout(v_P, opts['in_keep_prob'])
        print('v_P', v_P)

        print('Self-Matching Attention')
        SM_star = []
        for t in range(opts['p_length']):
            # Calculate s_t
            W_p1_v_P = self.mat_weight_mul(v_P, self.W_smP1)  # [batch_size, p_length, state_size]
            tiled_v_tP = tf.concat([tf.reshape(v_P[:, t, :], [opts['batch_size'], 1, -1])] * opts['p_length'], 1)
            W_p2_v_tP = self.mat_weight_mul(tiled_v_tP, self.W_smP2)

            tanh = tf.tanh(W_p1_v_P + W_p2_v_tP)
            s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_SM, [-1, 1])))
            a_t = tf.nn.softmax(s_t, 1)
            tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * opts['state_size'], 2)
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, v_P), 1)  # [batch_size, 2 * state_size]

            # gate
            v_tP_c_t = tf.expand_dims(tf.concat([tf.squeeze(v_P[:, t, :]), c_t], 1), 1)
            g_t = tf.sigmoid(self.mat_weight_mul(v_tP_c_t, self.W_g_SM))
            v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t))
            SM_star.append(v_tP_c_t_star)
        SM_star = tf.stack(SM_star, 1)
        unstacked_SM_star = tf.unstack(SM_star, opts['p_length'], 1)
        with tf.variable_scope('Self_match') as scope:
            SM_fw_cell = self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob'])
            SM_bw_cell = self.DropoutWrappedGRUCell(opts['state_size'], opts['in_keep_prob'])
            SM_outputs, SM_final_fw, SM_final_bw = tf.contrib.rnn.static_bidirectional_rnn(SM_fw_cell, SM_bw_cell,
                                                                                           unstacked_SM_star,
                                                                                           dtype=tf.float32)
            h_P = tf.stack(SM_outputs, 1)
        h_P = tf.nn.dropout(h_P, opts['in_keep_prob'])
        print('h_P', h_P)

        print('Output Layer')
        # calculate r_Q
        W_ruQ_u_Q = self.mat_weight_mul(u_Q, self.W_ruQ)  # [batch_size, q_length, 2 * state_size]
        W_vQ_V_rQ = tf.matmul(self.W_VrQ, self.W_vQ)
        W_vQ_V_rQ = tf.stack([W_vQ_V_rQ] * opts['batch_size'], 0)  # stack -> [batch_size, state_size, state_size]

        tanh = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ)
        s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_rQ, [-1, 1])))
        a_t = tf.nn.softmax(s_t, 1)
        tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'], 2)
        r_Q = tf.reduce_sum(tf.multiply(tiled_a_t, u_Q), 1)  # [batch_size, 2 * state_size]
        r_Q = tf.nn.dropout(r_Q, opts['in_keep_prob'])
        print('r_Q', r_Q)

        # r_Q as initial state of ans ptr
        h_a = None
        p = [None for _ in range(2)]
        for t in range(2):
            W_hP_h_P = self.mat_weight_mul(h_P, self.W_hP)  # [batch_size, p_length, state_size]

            if t == 0:
                h_t1a = r_Q
            else:
                h_t1a = h_a
            print('h_t1a', h_t1a)
            tiled_h_t1a = tf.concat([tf.reshape(h_t1a, [opts['batch_size'], 1, -1])] * opts['p_length'], 1)
            W_ha_h_t1a = self.mat_weight_mul(tiled_h_t1a, self.W_ha)

            tanh = tf.tanh(W_hP_h_P + W_ha_h_t1a)
            s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(self.B_v_ap, [-1, 1])))
            a_t = tf.nn.softmax(s_t, 1)
            p[t] = a_t

            tiled_a_t = tf.concat([tf.reshape(a_t, [opts['batch_size'], -1, 1])] * 2 * opts['state_size'], 2)
            c_t = tf.reduce_sum(tf.multiply(tiled_a_t, h_P), 1)  # [batch_size, 2 * state_size]

            if t == 0:
                AnsPtr_state = self.AnsPtr_cell.zero_state(opts['batch_size'], dtype=tf.float32)
                h_a, _ = self.AnsPtr_cell(c_t, (AnsPtr_state, r_Q))
                h_a = h_a[1]
                print(h_a)
        print(p)
        p1 = p[0]
        p2 = p[1]

        answer_si_idx = answer_si
        answer_ei_idx = answer_ei

        """	
        ce_si = tf.nn.softmax_cross_entropy_with_logits(labels = answer_si, logits = p1)
        ce_ei = tf.nn.softmax_cross_entropy_with_logits(labels = answer_ei, logits = p2)
        print(ce_si, ce_ei)
        loss_si = tf.reduce_sum(ce_si)
        loss_ei = tf.reduce_sum(ce_ei)
        loss = loss_si + loss_ei
        """

        batch_idx = tf.reshape(tf.range(0, opts['batch_size']), [-1, 1])
        answer_si_re = tf.reshape(answer_si_idx, [-1, 1])
        batch_idx_si = tf.concat([batch_idx, answer_si_re], 1)
        answer_ei_re = tf.reshape(answer_ei_idx, [-1, 1])
        batch_idx_ei = tf.concat([batch_idx, answer_ei_re], 1)

        log_prob = tf.multiply(tf.gather_nd(p1, batch_idx_si), tf.gather_nd(p2, batch_idx_ei))
        loss = -tf.reduce_sum(tf.log(log_prob + 0.0000001))

        return loss, loss, loss