import tensorflow as tf
from MNIST.config import FLAGS


class CNN(object):
    def __init__(self,
                 ):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None,
                                                               FLAGS.input_size,
                                                               FLAGS.input_size,
                                                               FLAGS.channel], name='input-x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_labels], name='input-y')
        self.dropout_keep_pro = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')
        with tf.variable_scope('layer1-conv1'):
            conv1_weight = tf.get_variable(name='conv1',
                                           shape=[FLAGS.conv1_size,
                                                  FLAGS.conv1_size,
                                                  FLAGS.channel,
                                                  FLAGS.conv1_deep],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable(name='biases',
                                           shape=[FLAGS.conv1_deep],
                                           initializer=tf.constant_initializer(0.1))
            conv1 = tf.nn.conv2d(self.input_x, conv1_weight, strides=FLAGS.conv1_strides, padding=FLAGS.conv1_padding)
            # print(conv1)  # 28 * 28  * 32
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        with tf.name_scope('layer2-pool1'):
            pool1 = tf.nn.max_pool(value=relu1, ksize=FLAGS.pool1_ksize,
                                   strides=FLAGS.pool1_strides, padding='SAME')
            # print(pool1)  # 14*14*32
        with tf.variable_scope('layer3-conv2'):
            conv2_weights = tf.get_variable(name='weight',
                                            shape=[FLAGS.conv2_size,
                                                   FLAGS.conv2_size,
                                                   FLAGS.conv1_deep,
                                                   FLAGS.conv2_deep],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable(name='bias',
                                           shape=[FLAGS.conv2_deep],
                                           initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.conv2d(
                input=pool1, filter=conv2_weights, strides=FLAGS.conv2_strides, padding=FLAGS.conv2_padding)

            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        with tf.name_scope('layer4-pool2'):
            pool2 = tf.nn.max_pool(
                value=relu2, ksize=FLAGS.pool2_ksize,
                strides=FLAGS.pool2_strides, padding='SAME')
        pool_shape = pool2.get_shape()
        # print('---------', pool_shape)
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(tensor=pool2, shape=[-1, nodes])
        # reshaped = tf.reshape(tensor=pool2, shape=[pool_shape[0], nodes])

        with tf.variable_scope('layer5-fc1'):
            fc1_weights = tf.get_variable(
                name='weight', shape=[nodes, FLAGS.fc1_size],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc1_biases = tf.get_variable(
                name='bias', shape=[FLAGS.fc1_size], initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            fc1_l2_loss = tf.nn.l2_loss(fc1_weights)
            fc1 = tf.nn.dropout(fc1, self.dropout_keep_pro)

        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable(
                name='weight', shape=[FLAGS.fc2_size, FLAGS.num_labels],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc2_biases = tf.get_variable(
                name='bias', shape=[FLAGS.num_labels], initializer=tf.constant_initializer(0.1))
            fc2_l2_loss = tf.nn.l2_loss(fc2_weights)
            self.logits = tf.matmul(fc1, fc2_weights) + fc2_biases

        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=tf.argmax(self.input_y, axis=1)
            )
            self.loss = tf.reduce_mean(losses) + FLAGS.l2_regul_rate * (fc1_l2_loss + fc2_l2_loss)

        with tf.name_scope('pre-acc'):
            self.predictions = tf.argmax(self.logits, axis=1, name='prediction')
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


if __name__ == '__main__':
    print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
    for attr, value in sorted(FLAGS.__flags.items()):
        print('{}={}'.format(attr.upper(), value))
