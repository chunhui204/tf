def _batch_norm(x, istraining):
    #定义beta, gamma，是通过训练更新的参数，trainable=True
    gamma = _get_variable('gamma', 
                          shape=(x.shape[-1]), 
                          initializer=tf.constant_initializer(1.0),
                          trainable = True)
    beta = _get_variable("beta",
                         shape=(x.shape[-1]),
                         initializer=tf.constant_initializer(0.0),
                         trainable = True)
    #定义moving_mean, moving_variance,是testing中使用的mean, var，是在训练中通过moving average更新的，note:不参与训练
    moving_mean = _get_variable('moving_mean',
                                shape=(x.shape[-1]),
                                initializer=tf.constant_initializer(0.0),
                                trainable = False)
    moving_variance = _get_variable('moving_variance',
                                shape=(x.shape[-1]),
                                initializer=tf.constant_initializer(1.0),
                                trainable = False)
    def bn_tarining():
        mean, variance = tf.nn.moments(x, axes=[0,1,2])
        #moving average具体操作，定义依赖关系，先更新moving变量再求这层的bn
        update_moving_mean = tf.assign(moving_mean, BN_DECAY*moving_mean + (1-BN_DECAY)*mean)
        update_moving_variance = tf.assign(moving_variance, BN_DECAY*moving_variance + (1-BN_DECAY)*variance)
        #training中使用的是该层实际的mean，var
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.nn.batch_normalization(x, beta, gamma, mean, variance, BN_EPSILON)
    def bn_testing():
        #testing，直接使用training阶段通过moving average更新的mean，var
        return tf.nn.batch_normalization(x, beta, gamma, moving_mean, moving_variance, BN_EPSILON)
    
    out = tf.cond(istraining, bn_tarining, bn_testing)
    
    return out
