import tensorflow as tf
from MNIST.config import FLAGS
from MNIST.cnn import CNN
from MNIST.data_helper import gen_test_data
cnn = CNN()

with tf.Session() as sess:
    x,y_test = gen_test_data(FLAGS.test_batch_size)
    x_test = sess.run(x)
    saver = tf.train.Saver()
    path = FLAGS.model_save_path + FLAGS.restore_model_name
    saver.restore(sess, save_path=path)
    feed_dic = {cnn.input_x: x_test, cnn.input_y: y_test, cnn.dropout_keep_pro: 1.0}
    acc = sess.run(cnn.accuracy, feed_dict=feed_dic)
    print('----acc:{}---'.format(acc))

