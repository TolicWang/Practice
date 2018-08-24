def gen_batch(x_train, y_train, begin, batch_size):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    x = x_train[start:end]
    y = y_train[start:end]
    return x, y