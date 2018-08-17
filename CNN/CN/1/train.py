import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from tensorflow.contrib import learn
from text_cnn import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定一个GPU
# parameters

# Data loadding  params
tf.flags.DEFINE_float(flag_name='dev_sample_percentage', default_value=0.1,
                      docstring='Percentage of the training data to use for validation')
tf.flags.DEFINE_string(flag_name='positive_data_file', default_value='../data/ham_5000.utf8',
                       docstring='positive data')
tf.flags.DEFINE_string(flag_name='negative_data_file', default_value='../data/spam_5000.utf8',
                       docstring='negative data')

# Model hyperparams
tf.flags.DEFINE_integer(flag_name='embedding', default_value=128, docstring='dimensionality of word')
tf.flags.DEFINE_string(flag_name='filter_size', default_value='3,4,5', docstring='filter size ')
tf.flags.DEFINE_integer(flag_name='num_filters', default_value=128, docstring='deep of filters')
tf.flags.DEFINE_float(flag_name='dropout', default_value=0.5, docstring='Drop out')
tf.flags.DEFINE_float(flag_name='L2_reg_lambda', default_value=0.0, docstring='L2')

# Training params
tf.flags.DEFINE_integer(flag_name='batch_size', default_value=32, docstring='batch size')
tf.flags.DEFINE_float(flag_name='learning_rate', default_value=0.1, docstring='learning rate')

tf.flags.DEFINE_boolean(flag_name='allow_soft_placement', default_value='True',
                        docstring='allow_soft_placement')  # 找不到指定设备时，是否自动分配
tf.flags.DEFINE_boolean(flag_name='log_device_placement', default_value='False',
                        docstring='log_device_placement ')  # 是否打印配置日志

FLAGS = tf.flags.FLAGS
# FLAGS.flag_values_dict()  # 解析参数成字典
FLAGS._parse_flags()
print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))

# Load data
x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# max_document_length = max([len(x.split(' ')) for x in x_text])
max_document_length = 120
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))  # 得到每个样本中，每个单词对应在词典中的序号

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]  # 打乱数据
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]  # 划分训练集和验证集
#

print(x_train.shape)
print(len(vocab_processor.vocabulary_))
print('train/dev split:{:d}/{:d}'.format(len(y_train), len(y_dev)))
#
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.6
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      vocab_size=len(vocab_processor.vocabulary_),
                      embedding_size=FLAGS.embedding,
                      filter_sizes=list(map(int, FLAGS.filter_size.split(','))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.L2_reg_lambda
                      )
        global_step = tf.Variable(0, trainable=False)
        with tf.device('/gpu:0'):
            # train_step = tf.train.GradientDescentOptimizer(
            #     FLAGS.learning_rate).minimize(loss=cnn.loss, global_step=global_step)
            train_step = tf.train.AdamOptimizer(1e-3).minimize(loss=cnn.loss, global_step=global_step)
    sess.run(tf.global_variables_initializer())
    last = datetime.datetime.now()
    for i in range(100000):
        x, y = data_helper.gen_batch(x_train, y_train, i, FLAGS.batch_size)
        feed_dic = {cnn.input_x: x, cnn.input_y: y, cnn.dropout_keep_prob: FLAGS.dropout}
        _, loss, acc = sess.run([train_step, cnn.loss, cnn.accuracy], feed_dict=feed_dic)

        if (i % 50) == 0:
            now = datetime.datetime.now()
            print('loss:{},acc:{}---time:{}'.format(loss, acc, now - last))
            last = now
        if (i % 1000 == 0):
            feed_dic = {cnn.input_x: x_dev, cnn.input_y: y_dev, cnn.dropout_keep_prob: 1.0}
            _, loss, acc = sess.run([train_step, cnn.loss, cnn.accuracy], feed_dict=feed_dic)
            print('-------------loss:{},acc:{}---time:{}--step:{}'.format(loss, acc, now - last, i))
