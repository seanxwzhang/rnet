class RNet:
    
    def __init__(self, options):
        self.options = options
        with tf.device('/cpu:0'):
            # Weights
            pass
    
    def build_model(self):
        options = self.options
        # placeholders
        batch_size = options['batch_size']
        p_length = options['p_length']
        q_length = options['q_length']
        emb_dim = options['emb_dim']

        eP = tf.placeholder(tf.float32, [batch_size, p_length, emb_dim])
        eQ = tf.placeholder(tf.float32, [batch_size, q_length, emb_dim])
        asi = tf.placeholder(tf.float32, [batch_size, p_length])
        aei = tf.placeholder(tf.float32, [batch_size, p_length])

        print('Shape of eP: {}'.format(eP.get_shape()))
        print('Shape of eQ: {}'.format(eQ.get_shape()))
        print('Shape of asi: {}'.format(asi.get_shape()))
        print('Shape of aei: {}'.format(aei.get_shape()))
        
        # embeddings concatenation
        # TODO: add character level embeddings
        eQcQ = eQ
        ePcP = eP

        # unstack here because stack_bidirectional_rnn requires
        # input as list of tensors
        unstacked_eQcQ = tf.unstack(eQcQ, q_length, 1)
        unstacked_ePcP = tf.unstack(ePcP, p_length, 1)
        
        # Question and Passage Encoding
        # TODO: test dynamic bi-rnn, w/o stack

        ## difference: # of GRU layers, dropout application
        h_size = options['h_size']
        in_keep_prob = options['in_keep_prob']
        with tf.variable_scope('encoding') as scope:
            gru_cells_fw = [tf.contrib.rnn.GRUCell(h_size) for _ in range(1)]
            gru_cells_bw = [tf.contrib.rnn.GRUCell(h_size) for _ in range(1)]
            q_enc_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(
                gru_cells_fw, gru_cells_bw, unstacked_eQcQ, dtype=tf.float32, scope='context_encoding')
            tf.get_variable_scope().reuse_variables()
            p_enc_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(
                gru_cells_fw, gru_cells_bw, unstacked_ePcP, dtype=tf.float32, scope='context_encoding')
            uQ = tf.stack(q_enc_outputs, 1)
            uP = tf.stack(p_enc_outputs, 1)
            uQ = tf.nn.dropout(uQ, in_keep_prob)
            uP = tf.nn.dropout(uP, in_keep_prob)
            print('Shape of uP: {}'.format(uP.get_shape()))
            print('Shape of uQ: {}'.format(uQ.get_shape()))


