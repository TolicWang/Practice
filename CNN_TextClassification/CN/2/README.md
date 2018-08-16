中文垃圾邮件文本分类<br><br>
**注意：** 如果以下两个参数发生了改变，请删除`embedding.model`从新训练词向量，不然会报错<br>
```python
tf.flags.DEFINE_integer(flag_name='embedding_size', default_value=50, docstring='dimensionality of word')
tf.flags.DEFINE_integer(flag_name='padding_size', default_value=100, docstring='padding seize of eatch sample')
```