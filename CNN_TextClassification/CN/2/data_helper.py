from gensim.models.word2vec import Word2Vec
import gensim
import jieba
import re
import numpy as np
import os


def clean_str(string):
    string.strip('\n')
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def cut_line(line):
    line = clean_str(line)
    seg_list = jieba.cut(line)
    cut_words = " ".join(seg_list)
    return cut_words


def load_data_and_labels(positive_data_file, negative_data_file):
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

