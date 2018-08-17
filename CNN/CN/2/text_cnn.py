import tensorflow as tf


class TextCNN(object):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name='x-input')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='y-input')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        l2_loss = tf.constant(0.0)

        self.embedded_chars = self.input_x
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='kernel')
                b = tf.Variable(tf.constant(value=0.1, shape=[num_filters], name='bias'))
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    [1, 1, 1, 1],
                                    "VALID",
                                    name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h,
                                        [1, sequence_length - filter_size + 1, 1, 1],
                                        [1, 1, 1, 1],
                                        "VALID")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(values=pooled_outputs, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope('out-fc'):
            W = tf.get_variable('w',
                                shape=[num_filters_total, num_classes],
                                initializer=tf.truncated_normal_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='bias'))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.logits = tf.nn.xw_plus_b(h_drop, W, b, name='logits')
            self.predictions = tf.argmax(self.logits, 1, name='prediction')
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=tf.argmax(self.input_y, 1))
            self.loss = tf.reduce_mean(losses) + l2_loss * l2_reg_lambda
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
