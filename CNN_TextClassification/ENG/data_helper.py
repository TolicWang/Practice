import numpy as np
import re

from tensorflow.contrib import learn
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\,\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    positive = open(positive_data_file, 'rb').read().decode('utf-8')  # 得到一个字符串，因为含有中文所以加decode('utf-8')
    negative = open(negative_data_file, 'rb').read().decode('utf-8')

    positive_examples = positive.split('\n')[:-1]  # 将整个文本用换行符分割成一个一个的邮件
    negative_examples = negative.split('\n')[:-1]  # 得到的是一个list,list中的每个元素都是一封邮件（并且去掉最后一个换行符）

    positive_examples = [s.strip() for s in positive_examples]  # 去掉每个邮件开头和结尾的的空格
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples  # 两个列表相加构数据集
    x_text = [clean_str(sent) for sent in x_text]  # 去除每个邮件中的标点等无用的字符

    positive_label = [[0, 1] for _ in positive_examples]  # 构造one-hot 标签[[0, 1], [0, 1], [0, 1], [0, 1],....]
    negative_label = [[1, 0] for _ in negative_examples]

    # print(positive_label[:5])
    y = np.concatenate([positive_label, negative_label], axis=0)

    return [x_text, y]


def gen_batch(x_train, y_train, begin, batch_size):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    x = x_train[start:end]
    y = y_train[start:end]
    return x, y
x_text,y=load_data_and_labels('./data/rt-polarity.pos','./data/rt-polarity.neg')
max_document_length = max([len(x.split(' ')) for x in x_text])
# print(max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))  # 得到每个样本中，每个单词对应在词典中的序号
print(x[:3])
print(y[:3])
len(vocab_processor.vocabulary_)