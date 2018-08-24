import tensorflow as tf
from MNIST.config import FLAGS
from MNIST.cnn import CNN
import datetime
import os
from MNIST.data_helper import gen_batch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定一个GPU
print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('----------------Parameters--------------\n')

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.6
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN()
        global_step = tf.Variable(0, trainable=False)
        with tf.device('/gpu:0'):
            train_step = tf.train.GradientDescentOptimizer(
                FLAGS.learning_rate).minimize(loss=cnn.loss, global_step=global_step)
            # train_step = tf.train.AdamOptimizer(1e-3).minimize(loss=cnn.loss, global_step=global_step)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    last = datetime.datetime.now()
    for i in range(500000):
        batch_x, batch_y = gen_batch(FLAGS.batch_size)
        x = sess.run(batch_x)
        feed_dic = {cnn.input_x: x, cnn.input_y: batch_y,
                    cnn.dropout_keep_pro: FLAGS.dropout_keep_pro}
        _, loss, acc = sess.run([train_step, cnn.loss, cnn.accuracy], feed_dict=feed_dic)

        if (i % 100) == 0:
            now = datetime.datetime.now()
            print('loss:{},acc on train :{}---time:{}'.format(loss, acc, now - last))
            last = now
        if (i % FLAGS.save_freq) == 0:
            print('Iterations:  ',i)
            saver.save(sess, os.path.join(FLAGS.model_save_path,
                                          FLAGS.model_name),
                       global_step=global_step, write_meta_graph=False)
