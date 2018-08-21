from text_cnn import TextCNN
from config import FLAGS
import tensorflow as tf
import data_helper

x_test_data, y_test = data_helper.load_data_and_labels(FLAGS.test_data_file, FLAGS.test_label_file)

padded_sentences_test, max_padding_length = data_helper.padding_sentence(
    sentences=x_test_data,
    padding_sentence_length=FLAGS.padding_sentence_length,
    padding_move=FLAGS.padding_move)

x_test, vocabulary_len = data_helper.embedding_sentences(
    embedding_file=FLAGS.embedding_file, padded_sentences=padded_sentences_test,
    embedding_dimension=FLAGS.embedding_dimension)

print("x_test.shape = {}".format(x_test.shape))
print("y_test.shape = {}".format(y_test.shape))

cnn = TextCNN(sequence_length=FLAGS.padding_sentence_length,
              num_classes=FLAGS.num_classes,
              embedding_dimension=FLAGS.embedding_dimension,
              filter_sizes=list(map(int, FLAGS.filter_size.split(','))),
              num_filters=FLAGS.num_filters,
              l2_reg_lambda=FLAGS.L2_reg_lambda
              )

with tf.Session() as sess:
    saver = tf.train.Saver()
    path = FLAGS.model_save_path + FLAGS.restore_model_name
    saver.restore(sess, save_path=path)
    feed_dic = {cnn.input_x: x_test, cnn.input_y: y_test, cnn.dropout_keep_prob: FLAGS.dropout}
    acc = sess.run(cnn.accuracy, feed_dict=feed_dic)
    print('----acc:{}---'.format(acc))
