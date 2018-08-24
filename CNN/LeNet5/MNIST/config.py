import tensorflow as tf

# input size
tf.flags.DEFINE_integer(flag_name='input_size', default_value=28, docstring='input_size')
tf.flags.DEFINE_integer(flag_name='channel', default_value=1, docstring='input_channel')

#   layer1-conv1 parameters
tf.flags.DEFINE_integer(flag_name='conv1_size', default_value=5, docstring='conv1_size')
tf.flags.DEFINE_integer(flag_name='conv1_deep', default_value=32, docstring='conv1_deep')
tf.flags.DEFINE_string(flag_name='conv1_padding', default_value='SAME', docstring='conv1_padding')
tf.flags.DEFINE_string(flag_name='conv1_strides', default_value=[1, 1, 1, 1], docstring='conv1_strides')

#   layer2-max pool parameters
tf.flags.DEFINE_string(flag_name='pool1_ksize', default_value=[1, 2, 2, 1], docstring='pool1_ksize')
tf.flags.DEFINE_string(flag_name='pool1_strides', default_value=[1, 2, 2, 1], docstring='pool1_strides')

#   layer3-conv2 parameters
tf.flags.DEFINE_integer(flag_name='conv2_size', default_value=5, docstring='conv2_size')
tf.flags.DEFINE_integer(flag_name='conv2_deep', default_value=64, docstring='conv2_deep')
tf.flags.DEFINE_string(flag_name='conv2_padding', default_value='SAME', docstring='conv2_padding')
tf.flags.DEFINE_string(flag_name='conv2_strides', default_value=[1, 1, 1, 1], docstring='conv2_strides')

#   layer4-maxpool2 parameters
tf.flags.DEFINE_string(flag_name='pool2_ksize', default_value=[1, 2, 2, 1], docstring='pool2_ksize')
tf.flags.DEFINE_string(flag_name='pool2_strides', default_value=[1, 2, 2, 1], docstring='pool2_strides')

#   layer5-full connection parameters
tf.flags.DEFINE_integer(flag_name='fc1_size', default_value=512, docstring='fc1_size')

#   layer6-full connection parameters
tf.flags.DEFINE_integer(flag_name='fc2_size', default_value=512, docstring='fc2_size')
tf.flags.DEFINE_integer(flag_name='num_labels', default_value=10, docstring='num_labels')

#   training parameters
tf.flags.DEFINE_integer(flag_name='training_ite', default_value=800000, docstring='training_ite')
tf.flags.DEFINE_boolean(flag_name='allow_soft_placement', default_value='True',
                        docstring='allow_soft_placement')  # 找不到指定设备时，是否自动分配
tf.flags.DEFINE_float(flag_name='learning_rate', default_value=0.01, docstring='learning_rate')
tf.flags.DEFINE_integer(flag_name='batch_size', default_value=32, docstring='batch_size')
tf.flags.DEFINE_float(flag_name='l2_regul_rate', default_value=0.0001, docstring='l2_regul_rate')
tf.flags.DEFINE_float(flag_name='dropout_keep_pro', default_value=0.5, docstring='drop_keep_pro')

tf.flags.DEFINE_string(flag_name='model_save_path', default_value='./model/', docstring='model_save_path')
tf.flags.DEFINE_string(flag_name='model_name', default_value='model.ckpt', docstring='model_name')
tf.flags.DEFINE_string(flag_name='restore_model_name', default_value='model.ckpt-1001', docstring='restore_model_name')
tf.flags.DEFINE_integer(flag_name='save_freq', default_value=5000, docstring='save_freq')

tf.flags.DEFINE_integer(flag_name='test_batch_size', default_value=5000, docstring='test_batch_size')

FLAGS = tf.flags.FLAGS
# FLAGS.flag_values_dict()  # 解析参数成字典
FLAGS._parse_flags()
# print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
# for attr, value in (FLAGS.__flags.items()):
#     print('{}={}'.format(attr.upper(), value))
