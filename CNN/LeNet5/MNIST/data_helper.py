from tensorflow.examples.tutorials.mnist import input_data
from MNIST.config import FLAGS
import tensorflow as tf

mnist = input_data.read_data_sets("./data/MNIST_data", one_hot=True)


def gen_batch(batch_size):
    x, y = mnist.train.next_batch(batch_size)
    reshped_x = tf.reshape(x,
                           [batch_size,
                            FLAGS.input_size,
                            FLAGS.input_size,
                            FLAGS.channel])
    return reshped_x, y


def gen_test_data(batch_size):
    x = mnist.test.images[:batch_size]
    y = mnist.test.labels[:batch_size]
    reshped_x = tf.reshape(x,
                           [len(y),
                            FLAGS.input_size,
                            FLAGS.input_size,
                            FLAGS.channel])
    return reshped_x, y

# reshped_x, y = gen_test_data()
# print(reshped_x)