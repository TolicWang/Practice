中文垃圾邮件文本分类<br><br>
**注意：** 如果以下两个参数发生了改变，请删除`embedding.model`从新训练词向量，不然会报错<br>
```python
tf.flags.DEFINE_integer(flag_name='embedding_size', default_value=50, docstring='dimensionality of word')
tf.flags.DEFINE_integer(flag_name='padding_size', default_value=100, docstring='padding seize of eatch sample')
```
总体上来说，效果比前一种方法好点。虽然准确率都能达到1，但速度相对要快点。<br>
当分类问题复杂一点的话，这种方法的优势会更明显。<br>

数据预处理完成之后，每封邮件等同于一个维度为`[100,50]`的二维矩阵，对应每个batch的维度就为 `[?,100,50,1]`<br>

**卷积网络的结构为：**<br>
第一层，卷积层：filter尺寸为 `[3,50,1,128]`，`[4,50,1,128]`，`[5,50,1,128]`；经过三个不同尺寸局卷积核卷积之后，三个部分对应的形状分别为`[98,1,1,128]`,`[97,1,1,128]`,`[96,1,1,128]`；
<br><br>
第二层，池化层：filter尺寸为 `[1,98,1,1]`，`[1,97,1,1]`，`[1,96,1,1]`；经过池化后，三个部分对应的形状分别`[1,1,1,128]`，`[1,1,1,128]`，`[1,1,1,128]`
<br><br>
第三层，全连接层；全连接层的节点个数为128*3
