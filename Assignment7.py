#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
get_ipython().system('ls /home/aistudio/work')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# ### 本地拟合结果
# ![](https://ai-studio-static-online.cdn.bcebos.com/1a9b0da9a5724de2adc8c95812f37921cf078702fe3c4cc6b85fc5425b06b315)
# 
# 
# ### 增加的Class
# ```python
# # Class 1.1 均匀分布(-0.1， 0.1)
# class SimpleLSTMRNNwithUniformDis(fluid.Layer):
# 
# # Class 1.2
# class SentimentClassifier(fluid.Layer):
# 
# # Class 2.1 高斯分布（0， 0.1）
# class SimpleLSTMRNNwithNormalDis(fluid.Layer):
# 
# # Class 2.2
# class SentimentClassifierwithNormalDis(fluid.Layer):
# ```
# 
# ### 参数修改
# ```python
# for i in range(self._num_layers):
#             weight_1 = self.create_parameter(
#                 attr=fluid.ParamAttr(
#                     initializer=fluid.initializer.NormalInitializer(
#                         loc=self._init_scale-0.1, scale=self._init_scale)),
#                 shape=[self._hidden_size * 2, self._hidden_size * 4],
#                 dtype="float32",
#                 default_initializer=fluid.initializer.NormalInitializer(
#                     loc=self._init_scale-0.1, scale=self._init_scale))
#             self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
#             bias_1 = self.create_parameter(
#                 attr=fluid.ParamAttr(
#                     initializer=fluid.initializer.NormalInitializer(
#                         loc=self._init_scale-0.1, scale=self._init_scale)),
#                 shape=[self._hidden_size * 4],
#                 dtype="float32",
#                 default_initializer=fluid.initializer.Constant(0.0))
# ```

# In[1]:


# 学习“百度架构师手把手教深度学习”中的课节5中的视频0304和0310，其中0304只要从43分45秒开始就行。
# 下面的程序为IMDB情绪分析的具体代码，请完成
# 1）LSTM的参数初始化方式对梯度收敛会造成影响，请将下面代码中LSTM模型的参数初始化方式从[-0.1,0.1]的均匀分布改为（0,0.1）的高斯分布，
#    并比较两者损失函数收敛谁更快
# 2）下面的程序没有对情感分析训练模型进行测试，请从batch中选择一批数据，打印其中每个样本，以及该样本的情感分析结果
# 3）修改程序，尽量使用GPU来完成（近期网上学习的人极具增加，导致GPU免费资源不够，若不能使用GPU环境，可能需要多次尝试）
import io
import os
import re
import sys
import six
import requests
import string
import tarfile
import hashlib
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
import matplotlib.pyplot as plt


def download():
    # 通过python的requests类，下载存储在
    # https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz的文件
    corpus_url = "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz"
    web_request = requests.get(corpus_url)
    corpus = web_request.content

    # 将下载的文件写在当前目录的aclImdb_v1.tar.gz文件内
    with open("./aclImdb_v1.tar.gz", "wb") as f:
        f.write(corpus)
    f.close()


def load_imdb(is_training):
    data_set = []

    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的rarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感

    for label in ["pos", "neg"]:
        with tarfile.open("./aclImdb_v1.tar.gz") as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training                 else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label))
                tf = tarf.next()

    return data_set


def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")

        data_set.append((sentence, sentence_label))

    return data_set


def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        sentence = [word2id_dict[word] if word in word2id_dict                         else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True):
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []

    if len(sentence_batch) == batch_size:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")

# Class 1.1 均匀分布(-0.1， 0.1)
class SimpleLSTMRNNwithUniformDis(fluid.Layer):

    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None):

        super(SimpleLSTMRNNwithUniformDis, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])

            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)
                gate_input = fluid.layers.elementwise_add(gate_input, bias)

                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)

                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)

                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m

                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')

            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))

        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])

        return real_res, last_hidden, last_cell

# Class 1.2
class SentimentClassifier(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 class_num=2,
                 num_layers=1,
                 num_steps=128,
                 init_scale=0.1,
                 dropout=None):
        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout

        self.simple_lstm_rnn = SimpleLSTMRNNwithUniformDis(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)

        self.embedding = Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.class_num],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.class_num],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label):
        init_hidden_data = np.zeros(
            (1, batch_size, embedding_size), dtype='float32')
        init_cell_data = np.zeros(
            (1, batch_size, embedding_size), dtype='float32')

        init_hidden = fluid.dygraph.to_variable(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = fluid.dygraph.to_variable(init_cell_data)
        init_cell.stop_gradient = True

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)

        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')

        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self.hidden_size])

        projection = fluid.layers.matmul(last_hidden, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.class_num])
        pred = fluid.layers.softmax(projection, axis=-1)

        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reduce_mean(loss)

        return pred, loss

# Class 2.1 高斯分布（0， 0.1）
class SimpleLSTMRNNwithNormalDis(fluid.Layer):

    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None):

        super(SimpleLSTMRNNwithNormalDis, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(
                        loc=self._init_scale-0.1, scale=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.NormalInitializer(
                    loc=self._init_scale-0.1, scale=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(
                        loc=self._init_scale-0.1, scale=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])

            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)
                gate_input = fluid.layers.elementwise_add(gate_input, bias)

                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)

                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)

                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m

                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')

            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))

        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])

        return real_res, last_hidden, last_cell

# Class 2.2
class SentimentClassifierwithNormalDis(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 class_num=2,
                 num_layers=1,
                 num_steps=128,
                 init_scale=0.1,
                 dropout=None):
        super(SentimentClassifierwithNormalDis, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout

        self.simple_lstm_rnn = SimpleLSTMRNNwithNormalDis(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)

        self.embedding = Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.class_num],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.class_num],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label):
        init_hidden_data = np.zeros(
            (1, batch_size, embedding_size), dtype='float32')
        init_cell_data = np.zeros(
            (1, batch_size, embedding_size), dtype='float32')

        init_hidden = fluid.dygraph.to_variable(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = fluid.dygraph.to_variable(init_cell_data)
        init_cell.stop_gradient = True

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)

        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')

        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self.hidden_size])

        projection = fluid.layers.matmul(last_hidden, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.class_num])
        pred = fluid.layers.softmax(projection, axis=-1)

        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reduce_mean(loss)

        return pred, loss




# 下载数据集
# download()
# 生成训练样本和测试样本集
train_corpus = load_imdb(True)
test_corpus = load_imdb(False)

for i in range(5):
    print("sentence %d, %s" % (i, train_corpus[i][0]))
    print("sentence %d, label %d" % (i, train_corpus[i][1]))
# 预处理
train_corpus = data_preprocess(train_corpus)
test_corpus = data_preprocess(test_corpus)
print(train_corpus[:5])

# 计算词频，生成词Id
word2id_freq, word2id_dict = build_dict(train_corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))

# 将句子转换为Id序列
train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
print("%d tokens in the corpus" % len(train_corpus))
print(train_corpus[:5])
print(test_corpus[:5])

# 生成mini-batch
for _, batch in zip(range(10), build_batch(word2id_dict,
                                           train_corpus, batch_size=3, epoch_num=3, max_seq_len=30)):
    print(batch)

# 开始训练
batch_size = 64
epoch_num = 5
embedding_size = 128
step = 0
learning_rate = 0.01
max_seq_len = 128

# GPU Flag here
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
with fluid.dygraph.guard(place=place):
    # 创建一个用于情感分类的网络实例，sentiment_classifier
    sentiment_classifier = SentimentClassifier(
        embedding_size, vocab_size, num_steps=max_seq_len)
    # 创建优化器AdamOptimizer，用于更新这个网络的参数
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list=sentiment_classifier.parameters())
    steps1_x = []
    steps1_y = []
    for sentences, labels in build_batch(
            word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len):

        sentences_var = fluid.dygraph.to_variable(sentences)
        labels_var = fluid.dygraph.to_variable(labels)
        pred, loss = sentiment_classifier(sentences_var, labels_var)

        loss.backward()
        adam.minimize(loss)
        sentiment_classifier.clear_gradients()

        step += 1
        steps1_x.append(step)
        steps1_y.append(float(loss.numpy()[0]))
        if step % 10 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        if step == 1000:
            break


with fluid.dygraph.guard(place=place):
    sentiment_classifier = SentimentClassifierwithNormalDis(
        embedding_size, vocab_size, num_steps=max_seq_len)
    # 创建优化器AdamOptimizer，用于更新这个网络的参数
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list=sentiment_classifier.parameters())

    steps2_x = []
    steps2_y = []
    step = 0
    for sentences, labels in build_batch(
            word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len):

        sentences_var = fluid.dygraph.to_variable(sentences)
        labels_var = fluid.dygraph.to_variable(labels)
        pred, loss = sentiment_classifier(sentences_var, labels_var)

        loss.backward()
        adam.minimize(loss)
        sentiment_classifier.clear_gradients()
        steps2_x.append(step)
        steps2_y.append(float(loss.numpy()[0]))
        step += 1
        if step % 10 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        if step == 1000:
            break
        
            
print(steps1_y)
print(steps2_y)

fig, left_axis = plt.subplots()

p1, = left_axis.plot(steps1_x, steps1_y, 'ro-')
right_axis = left_axis.twinx()
p2, = right_axis.plot(steps2_x, steps2_y, 'bo-')
plt.xticks(steps1_x, steps2_x, rotation=0)  # 设置x轴的显示形式

# 设置左坐标轴以及右坐标轴的范围、精度
left_axis.set_ylim(0, 1.01)
left_axis.set_yticks(np.arange(0, 1.01, 0.1))
right_axis.set_ylim(0, 1.01)
right_axis.set_yticks(np.arange(0, 1.01, 0.1))

# 设置坐标及标题的大小、颜色
left_axis.set_title('Loss Rate of different param distribution method')
left_axis.set_xlabel('Steps')
left_axis.set_ylabel('Uniform Distribution', color='r')
left_axis.tick_params(axis='y', colors='r')
right_axis.set_ylabel('Normal Distribution', color='b')
right_axis.tick_params(axis='y', colors='b')
plt.show()

