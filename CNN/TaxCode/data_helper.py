from gensim.models.word2vec import Word2Vec
import gensim
import jieba
import re
import pandas as pd
import numpy as np
import os
import datetime


def is_contain_chinese(string):
    """
    该方法的作用是检查一个字符串中是否含有中文字符
    :param string: 字符串类型
    :return:
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def clean_str(string):
    """
    该函数的作用是:
                    #如果含有中文，去掉所有的非中文
                    #若不含中文，仅仅保留英文字母
    exampel:
        a = "I am 诗仙李白。*&"   #诗仙李白
        b = "Do you love me?'&*^(*^(’。*&"  #do you love me

    :param string: 字符串类型
    :return: 返回处理后的字符串
    """
    string.strip('\n')
    if is_contain_chinese(string):
        string = re.sub("[^\u4e00-\u9fff]", " ", string)
    else:
        string = re.sub("[^A-Za-z]", " ", string).lower()
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def cut_line(line):
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line: 每一行都是字符串类型
    :return: 分词后的结果，如 ：     "衣带  渐宽  终  不悔"
    """
    line = clean_str(line)
    seg_list = jieba.cut(line)
    cut_words = " ".join(seg_list)
    return cut_words


def load_data_and_labels(data_file, label_file):
    """
    :param data_file: 文件格式如下：
    ---------------data.txt------------
    云服务费
    住宿费
    租赁服务费
    塔吊费用
    ------------------------------------
    :param label_file:文件格式如下：
    ---------------label.txt------------
    1
    3
    5
    6
    ------------------------------------
    :return:
    x_text:  [['云','服务费'],['住宿费'],['租赁','服务费'],['塔吊','费用']]
    y:       [1  3  5  6]
    """
    print("[===========================================")
    print("------load_data_and_labels() begin!--------")
    print("--------", datetime.datetime.now().isoformat(sep='-'), "-------")
    x_text = []
    for line in open(data_file, encoding='utf-8'):
        x_text.append(cut_line(line).split())
    labels = pd.read_csv(label_file, names=['C'])
    y = np.array(labels['C'])
    print("x_test len = {:d},y len = {:d}".format(len(x_text), len(y)))
    print("===========================================]\n\n")
    return x_text, y


def padding_sentence(sentences, padding_token='UNK', padding_sentence_length=None):
    """
    该函数的作用是 按最大长度Padding样本
    :param sentences: [['今天','天气','晴朗'],['你','真','好']]
    :param padding_token: padding 的内容，默认为'UNK'
    :param padding_sentence_length: 以5为例
    :return: [['今天','天气','晴朗','UNK','UNK'],['你','真','好','UNK'，'UNK']]
    """
    print("[===========================================")
    print("----------padding_sentence() begin!--------")
    print("--------", datetime.datetime.now().isoformat(sep='-'), "-------")

    max_padding_length = padding_sentence_length if padding_sentence_length is not \
                                                    None else max([len(sentence) for sentence in sentences])
    for i, sentence in enumerate(sentences):
        if len(sentence) < max_padding_length:
            sentence.extend([padding_token] * (max_padding_length - len(sentence)))
        else:
            sentences[i] = sentence[:max_padding_length]
    print('sentences len={:d},max_padding_length={:d}'.format(len(sentence), max_padding_length))
    print("===========================================]\n\n")
    return sentences, max_padding_length


def word2vector(sentences, embedding_dimension=50, min_count=5, window=5,
                embedding_file='./embedding.model'):
    print('-------word2vector------------')
    train_model = Word2Vec(sentences=sentences, size=embedding_dimension,
                           min_count=min_count, window=window)
    train_model.save(embedding_file)
    return train_model


def embedding_sentences(embedding_file='./data/sgns.renmin.word',
                        padded_sentences=None,
                        embedding_dimension=50,
                        min_count=5,
                        window=5):
    """
    本函数的作用是将 分词后的文本转化为用词向量来表示
    如果有模型就载入，没有就利用Word2Vec训练
    :param embedding_file: 词向量模型 embedding_model
    你    0.1 0.2 0.3 0.4
    明天  0.3 0.5 0.2 0.1
    :param padded_sentences:[['你','明天,'UNK'],['明天','UNK','UNK']]
    :param embedding_dimension:  4
    :param min_count:
    :param window:
    :return:
    [[[0.1,0.2,0.3,0.4],[0.3,0.5,0.2,0.1],[0,0,0,0]]
     [[0.3,0.5,0.2,0.1],[0,0,0,0],[0,0,0,0]]]
     shape: (2,3,4)
    """
    print("[===========================================")
    print("--------embedding_sentence() begin!--------")
    print("--------", datetime.datetime.now().isoformat(sep='-'), "-------")

    if os.path.exists(path=embedding_file):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_file, binary=False)
        embedded_dimension = model.vector_size
    else:
        model = word2vector(sentences=padded_sentences,
                            embedding_dimension=embedding_dimension,
                            min_count=min_count,
                            window=window)
        embedded_dimension = embedding_dimension
    all_vectors = []
    embedding_unknown = [0 for i in range(embedded_dimension)]
    for sentence in padded_sentences:
        this_vector = []
        for word in sentence:
            if word in model.wv.vocab:
                this_vector.append(model[word])
            else:
                this_vector.append(embedding_unknown)
        all_vectors.append(this_vector)
    all_vectors = np.array(all_vectors)
    print("embedded sentence shape {}".format(all_vectors.shape))
    print("--------embedding_sentence() finished!--------")
    print("--------", datetime.datetime.now().isoformat(sep='-'), "-------")
    print("===========================================]\n\n")
    return all_vectors, len(model.wv.vocab)


def gen_batch(x_train, y_train, begin, batch_size):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    x = x_train[start:end]
    y = y_train[start:end]
    return x, y


if __name__ == '__main__':
    data = './data/test_text.txt'
    label = './data/test_label.txt'
    xs, ys = load_data_and_labels(data, label)
    # padded_sentences, padded_length = padding_sentence(xs, padding_sentence_length=7)
    # x, vocabulary_len = embedding_sentences(padded_sentences=padded_sentences)
    # print(x.shape)

