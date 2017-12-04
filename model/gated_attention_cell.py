from tensorflow.python.ops import variable_scope as vs
class GatedAttentinoCell(RNNCell):
    
    def __init__(self, num_units, weights, reuse=None):
        
        super(GatedAttentinoCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self.WuQ = weights['WuQ'] # 2H * H
        self.WuP = weights['WuP'] # 2H * H
        self.WvP = weights['WvP'] # H * H
        
    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_units
    
    @staticmethod
    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert(mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def call(self, inputs, state):
        with vs.variable_scope('attention_pool'):
            utP = inputs # batch_size * 2H
            vtP = state
            WuQ_uQ = self.mat_weight_mul(uQ, self.WuQ) # batch_size x q_length x h_size
            WuP_utP = tf.matmul(utP, self.WuP) # batch_size x H
            WvP_vtP = tf.matmul(vtO, self.WvP)
            
            tanh = tf.tanh
