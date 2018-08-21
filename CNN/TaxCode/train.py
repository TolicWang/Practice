import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from tensorflow.contrib import learn
from text_cnn import TextCNN
from config import FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定一个GPU
print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
for attr, value in (FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))

# Load data and cut
x_train_data, y = data_helper.load_data_and_labels(FLAGS.train_data_file, FLAGS.train_label_file)


# Padding sentence
padded_sentences_train, max_padding_length = data_helper.padding_sentence(
    sentences=x_train_data, padding_sentence_length=FLAGS.padding_sentence_length)

x, vocabulary_len = data_helper.embedding_sentences(
    embedding_file=FLAGS.embedding_file, padded_sentences=padded_sentences_train,
    embedding_dimension=FLAGS.embedding_dimension)

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

print('--------------------------preProcess finished!-----------------------')
print('--------------------------preProcess finished!-----------------------')
print("vocabulary length={}".format(vocabulary_len))
print("x_train.shape = {}".format(x_train.shape))
print("y_train.shape = {}".format(y_train.shape))
print("x_dev.shape = {}".format(x_dev.shape))
print("y_dev.shape = {}".format(y_dev.shape))

print('train/dev split:{:d}/{:d}'.format(len(y_train), len(y_dev)))
# print(y_train[:100])
#


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.6
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length=FLAGS.padding_sentence_length,
                      num_classes=FLAGS.num_classes,
                      embedding_dimension=FLAGS.embedding_dimension,
                      filter_sizes=list(map(int, FLAGS.filter_size.split(','))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.L2_reg_lambda
                      )
        global_step = tf.Variable(0, trainable=False)
        with tf.device('/gpu:0'):
            train_step = tf.train.GradientDescentOptimizer(
                FLAGS.learning_rate).minimize(loss=cnn.loss, global_step=global_step)
            # train_step = tf.train.AdamOptimizer(1e-3).minimize(loss=cnn.loss, global_step=global_step)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    last = datetime.datetime.now()
    for i in range(FLAGS.training_ite):
        x, y = data_helper.gen_batch(x_train, y_train, i, FLAGS.batch_size)
        feed_dic = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: FLAGS.dropout}
        _, loss, acc = sess.run([train_step, cnn.loss, cnn.accuracy], feed_dict=feed_dic)
        print('-----------------',global_step)
        if (i % 100) == 0:
            now = datetime.datetime.now()
            print('loss:{},acc:{}---time:{}'.format(loss, acc, now - last))
            last = now
        if (i % 1000) == 0:
            feed_dic = {cnn.input_x: x_dev, cnn.input_y: y_dev, cnn.dropout_keep_prob: 1.0}
            _, loss, acc = sess.run([train_step, cnn.loss, cnn.accuracy], feed_dict=feed_dic)
            print('-------------loss:{},acc:{}---time:{}--step:{}'.format(loss, acc, now - last, i))
        if (i % FLAGS.save_freq) == 0:
            saver.save(sess, os.path.join(FLAGS.model_save_path,
                                          FLAGS.model_name),
                       global_step=global_step,)


