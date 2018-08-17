from gensim.models.word2vec import Word2Vec
import jieba
import re
import numpy as np
import os


def clean_str(string):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param string:
    :return: 返回处理后的字符串
    """
    string.strip('\n')
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def cut_line(line):
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line:
    :return: 分词后的结果，如 ：     衣带  渐宽  终  不悔
    """
    line = clean_str(line)
    seg_list = jieba.cut(line)
    cut_words = " ".join(seg_list)
    return cut_words


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    按行载入数据，然后分词。同时构造标签
    :param positive_data_file:
    :param negative_data_file:
    :return:  分词后的结果和标签
    x_text:   ['衣带 渐宽 终 不悔',' 为 伊 消得 人憔悴']
    y: [[1 0],[ 1 0]]
    """
    positive = []
    negative = []
    for line in open(positive_data_file, encoding='utf-8'):
        positive.append(cut_line(line).split())
    for line in open(negative_data_file, encoding='utf-8'):
        negative.append(cut_line(line).split())

    x_text = positive + negative

    positive_label = [[0, 1] for _ in positive]  # 构造one-hot 标签[[0, 1], [0, 1], [0, 1], [0, 1],....]
    negative_label = [[1, 0] for _ in negative]  # 构造one-hot 标签[[0, 1], [0, 1], [0, 1], [0, 1],....]
    y = np.concatenate([positive_label, negative_label], axis=0)

    return x_text, y


def padding_sentence(sentences, padding_token='UNK', padding_sentence_length=None):
    """
    该函数的作用是 按最大长度Padding样本
    :param sentences: [['今天','天气','晴朗'],['你','真','好']]
    :param padding_token: padding 的内容，默认为'UNK'
    :param padding_sentence_length: 以5为例
    :return: [['今天','天气','晴朗','UNK','UNK'],['你','真','好','UNK'，'UNK']]
    """
    max_padding_length = padding_sentence_length if padding_sentence_length is not \
                                                    None else max([len(sentence) for sentence in sentences])
    for i,sentence in enumerate(sentences):
        if len(sentence) < max_padding_length:
            sentence.extend([padding_token] * (max_padding_length - len(sentence)))
        else:
            sentences[i] = sentence[:max_padding_length]
    return sentences, max_padding_length


def word2vector(sentences, embedding_size=50, min_count=5, window=5,
                embedding_file='./embedding.model'):
    print('-------word2vector------------')
    train_model = Word2Vec(sentences=sentences, size=embedding_size,
                           min_count=min_count, window=window)
    train_model.save(embedding_file)
    return train_model


def embedding_sentences(embedding_file='./embedding.model',
                        padded_sentences=None,
                        embedding_size=50,
                        min_count=5,
                        window=5):
    """
    本函数的作用是将 分词后的文本转化为用词向量来表示
    如果有模型就载入，没有就利用Word2Vec训练
    :param embedding_file: 词向量模型 embedding_model
    你    0.1 0.2 0.3 0.4
    明天  0.3 0.5 0.2 0.1
    :param padded_sentences:[['你','太阳],['明天]]
    :param embedding_size:  4
    :param min_count:
    :param window:
    :return:
    [[0.1,0.2,0.3,0.4,0,0,0,0],[0.3,0.5,0.2,0.1,0,0,0,0]]
    """
    if os.path.exists(embedding_file):
        model = Word2Vec.load(embedding_file)
    else:
        model = word2vector(sentences=padded_sentences,
                            embedding_size=embedding_size,
                            min_count=min_count,
                            window=window)
    all_vectors = []
    embedding_unknown = [0 for i in range(embedding_size)]
    for sentence in padded_sentences:
        this_vector = []
        for word in sentence:
            if word in model.wv.vocab:
                this_vector.append(model[word])
            else:
                this_vector.append(embedding_unknown)
        all_vectors.append(this_vector)
    return all_vectors, len(model.wv.vocab)


def gen_batch(x_train, y_train, begin, batch_size):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    x = x_train[start:end]
    y = y_train[start:end]
    return x, y


if __name__ == '__main__':
    positive_data_file = '../data/ham_100.utf8'
    negative_data_file = '../data/ham_100.utf8'
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file)
    padded_sentences, max_padding_length = \
        padding_sentence(sentences=x_text, padding_sentence_length=100)
    embedded_sentences, vocabulary_len = embedding_sentences(padded_sentences=padded_sentences)
    x = np.array(embedded_sentences)
    print(x.shape)
    x_batch, y_batch = gen_batch(x, y, 2, 5)
    print(x_batch.shape)

