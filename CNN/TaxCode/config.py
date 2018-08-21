import tensorflow as tf

#   data path
tf.flags.DEFINE_string(flag_name='train_data_file', default_value='./data/train_text.txt',
                       docstring='train data')
tf.flags.DEFINE_string(flag_name='train_label_file', default_value='./data/train_label.txt',
                       docstring='train label')
tf.flags.DEFINE_string(flag_name='test_data_file', default_value='./data/test_text.txt',
                       docstring='test data')
tf.flags.DEFINE_string(flag_name='test_label_file', default_value='./data/test_label.txt',
                       docstring='test label')
tf.flags.DEFINE_string(flag_name='embedding_file', default_value='./data/sgns.merge.word',
                       docstring='test label')

#   data processing parameters
tf.flags.DEFINE_integer(flag_name='embedding_dimension', default_value=300, docstring='dimensionality of word')
tf.flags.DEFINE_integer(flag_name='padding_sentence_length', default_value=7, docstring='padding seize of eatch sample')
#   net work parameters
tf.flags.DEFINE_string(flag_name='filter_size', default_value='3,4,5', docstring='filter size ')
tf.flags.DEFINE_integer(flag_name='num_filters', default_value=128, docstring='deep of filters')
tf.flags.DEFINE_float(flag_name='dropout', default_value=0.5, docstring='Drop out')
tf.flags.DEFINE_float(flag_name='L2_reg_lambda', default_value=0.0, docstring='L2')
tf.flags.DEFINE_integer(flag_name='num_classes', default_value=3510, docstring='num_classes')

#   Training params
tf.flags.DEFINE_float(flag_name='dev_sample_percentage', default_value=0.1,
                      docstring='Percentage of the training data to use for validation')
tf.flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='batch size')
tf.flags.DEFINE_float(flag_name='learning_rate', default_value=0.001, docstring='learning rate')
tf.flags.DEFINE_boolean(flag_name='allow_soft_placement', default_value='True',
                        docstring='allow_soft_placement')  # 找不到指定设备时，是否自动分配
tf.flags.DEFINE_boolean(flag_name='log_device_placement', default_value='False',
                        docstring='log_device_placement ')  # 是否打印配置日志
tf.flags.DEFINE_integer(flag_name='training_ite', default_value=900000000, docstring='training_ite')

tf.flags.DEFINE_string(flag_name='model_save_path', default_value='./model/', docstring='model_save_path')
tf.flags.DEFINE_string(flag_name='model_name', default_value='model.ckpt', docstring='model_name')
tf.flags.DEFINE_integer(flag_name='save_freq',default_value=100000,docstring='save_freq')
FLAGS = tf.flags.FLAGS
# FLAGS.flag_values_dict()  # 解析参数成字典
FLAGS._parse_flags()