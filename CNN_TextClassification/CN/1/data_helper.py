import re
import jieba
import numpy as np
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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
    for line in open(positive_data_file,encoding='utf-8'):
        positive.append(cut_line(line))
    for line in open(negative_data_file,encoding='utf-8'):
        negative.append(cut_line(line))
    x_text = positive + negative

    positive_label = [[0, 1] for _ in positive]  # 构造one-hot 标签[[0, 1], [0, 1], [0, 1], [0, 1],....]
    negative_label = [[1, 0] for _ in negative]

    y = np.concatenate([positive_label, negative_label], axis=0)

    return x_text,y


def gen_batch(x_train, y_train, begin, batch_size):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    x = x_train[start:end]
    y = y_train[start:end]
    return x, y


if __name__ == '__main__':
    positive_data_file = '../data/ham_5000.utf8'
    negative_data_file = '../data/spam_5000.utf8'
    x_text,y=load_data_and_labels(positive_data_file, negative_data_file)
    print(x_text)



