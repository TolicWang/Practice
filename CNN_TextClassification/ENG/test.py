import numpy as np
import tensorflow as tf

W = np.random.random((5, 9))
input_x = [[1, 2, 3, 0], [2, 2, 3, 0]]
embedded_chars = tf.nn.embedding_lookup(W, input_x)
print(embedded_chars)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
print(embedded_chars_expanded)