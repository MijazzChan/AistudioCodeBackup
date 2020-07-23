#!/usr/bin/env python
# coding: utf-8

# # 第六次作业
# 2017326603075  陈浩骏  17信科1班
# ---
# ## 取距离词
# ```python
# def get_similar_tokens(query_token, k, embed):
#     W = embed.numpy()
#     x = W[word2id_dict[query_token]]
#     cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
#     flat = cos.flatten()
#     # 一维化向量后，利用argpartition和 argsort来找距离远近top_k，带权阵与标志词的向量点乘即为距离阵
#     indices_pos = np.argpartition(flat, -k)[-k:]
#     indices_neg = np.argpartition(flat, k)[-k:]
#     indices_pos = indices_pos[np.argsort(-flat[indices_pos])]
#     indices_neg = indices_neg[np.argsort(-flat[indices_neg])]
#     tmp_pos = []
#     tmp_neg = []
#     for i in indices_pos:
#         tmp_pos.append(str(id2word_dict[i]))
#     for i in indices_neg:
#         tmp_neg.append(str(id2word_dict[i]))
#     print('The nearest word of %s  -> %s' % (query_token, str(tmp_pos)))
#     print('The furthest word of %s -> %s' % (query_token, str(tmp_neg)))
# ```
# ## 取`text8.txt`的1,000,000长度的数据
# ---
# 结果：
# ```step 500, loss 0.638
# step 1000, loss 0.447
# Steps -> 1000 , Similar words:
# The nearest word of english  -> ['english', 'won', 'generally', 'person', 'major']
# The furthest word of english -> ['in', 'to', 'and', 'one', 'the']
# The nearest word of revolution  -> ['revolution', 'won', 'studies', 'also', 'alaska']
# The furthest word of revolution -> ['in', 'to', 'one', 'and', 'the']
# The nearest word of against  -> ['against', 'will', 'with', 'would', 'obligation']
# The furthest word of against -> ['in', 'to', 'and', 'the', 'one']
# step 1500, loss 0.432
# step 2000, loss 0.426
# Steps -> 2000 , Similar words:
# The nearest word of english  -> ['english', 'generally', 'them', 'made', 'subject']
# The furthest word of english -> ['and', 'in', 'the', 'broadcasting', 'attaining']
# The nearest word of revolution  -> ['revolution', 'states', 'united', 'won', 'about']
# The furthest word of revolution -> ['in', 'the', 'to', 'and', 'broadcasting']
# The nearest word of against  -> ['against', 'would', 'united', 'january', 'western']
# The furthest word of against -> ['in', 'and', 'the', 'broadcasting', 'attaining']
# step 2500, loss 0.452
# step 3000, loss 0.378
# Steps -> 3000 , Similar words:
# The nearest word of english  -> ['english', 'democratic', 'made', 'said', 'meet']
# The furthest word of english -> ['and', 'in', 'to', 'the', 'kenai']
# The nearest word of revolution  -> ['revolution', 'ocean', 'felt', 'entitled', 'divided']
# The furthest word of revolution -> ['kenai', 'broadcasting', 'attaining', 'in', 'the']
# The nearest word of against  -> ['against', 'goal', 'meet', 'book', 'strike']
# The furthest word of against -> ['the', 'kenai', 'attaining', 'broadcasting', 'ktva']
# step 3500, loss 0.354
# step 4000, loss 0.382
# Steps -> 4000 , Similar words:
# The nearest word of english  -> ['english', 'truly', 'quotes', 'wasn', 'pledge']
# The furthest word of english -> ['attaining', 'kenai', 'ktva', 'broadcasting', 'khmero']
# The nearest word of revolution  -> ['revolution', 'dues', 'customs', 'undefeated', 'seller']
# The furthest word of revolution -> ['attaining', 'kenai', 'broadcasting', 'solved', 'ktva']
# The nearest word of against  -> ['against', 'goal', 'strike', 'ut', 'phenomena']
# The furthest word of against -> ['attaining', 'kenai', 'broadcasting', 'the', 'ktva']
# step 4500, loss 0.327
# step 5000, loss 0.396
# Steps -> 5000 , Similar words:
# The nearest word of english  -> ['english', 'buffon', 'pledge', 'subfamilies', 'grassy']
# The furthest word of english -> ['attaining', 'kenai', 'broadcasting', 'solved', 'ktva']
# The nearest word of revolution  -> ['revolution', 'betrayal', 'dues', 'involve', 'propelled']
# The furthest word of revolution -> ['attaining', 'kenai', 'broadcasting', 'solved', 'in']
# The nearest word of against  -> ['against', 'federated', 'goal', 'orwell', 'strike']
# The furthest word of against -> ['kenai', 'attaining', 'broadcasting', 'and', 'the']
# step 5500, loss 0.394
# step 6000, loss 0.332
# Steps -> 6000 , Similar words:
# The nearest word of english  -> ['english', 'subfamilies', 'nathan', 'hippo', 'satirists']
# The furthest word of english -> ['attaining', 'kenai', 'broadcasting', 'ktva', 'solved']
# The nearest word of revolution  -> ['revolution', 'betrayal', 'neolithic', 'internationale', 'adventurer']
# The furthest word of revolution -> ['kenai', 'attaining', 'solved', 'broadcasting', 'ktva']
# The nearest word of against  -> ['against', 'refugees', 'strike', 'rushing', 'universes']
# The furthest word of against -> ['kenai', 'attaining', 'broadcasting', 'solved', 'ktva']
# step 6500, loss 0.301
# step 7000, loss 0.302
# Steps -> 7000 , Similar words:
# The nearest word of english  -> ['english', 'nathan', 'habitation', 'diffusing', 'nilo']
# The furthest word of english -> ['broadcasting', 'kenai', 'solved', 'attaining', 'and']
# The nearest word of revolution  -> ['revolution', 'betrayal', 'neolithic', 'propelled', 'ades']
# The furthest word of revolution -> ['kenai', 'solved', 'attaining', 'broadcasting', 'in']
# The nearest word of against  -> ['against', 'refugees', 'rushing', 'rebelled', 'strike']
# The furthest word of against -> ['kenai', 'attaining', 'solved', 'broadcasting', 'and']
# step 7500, loss 0.271
# step 8000, loss 0.244
# Steps -> 8000 , Similar words:
# The nearest word of english  -> ['english', 'burundi', 'habitation', 'satirists', 'nathan']
# The furthest word of english -> ['broadcasting', 'kenai', 'attaining', 'solved', 'ktva']
# The nearest word of revolution  -> ['revolution', 'betrayal', 'neolithic', 'glorified', 'stalinism']
# The furthest word of revolution -> ['kenai', 'broadcasting', 'in', 'attaining', 'solved']
# The nearest word of against  -> ['against', 'refugees', 'rushing', 'rebelled', 'clashed']
# The furthest word of against -> ['kenai', 'attaining', 'the', 'broadcasting', 'and']
# step 8500, loss 0.263
# step 9000, loss 0.280
# Steps -> 9000 , Similar words:
# The nearest word of english  -> ['english', 'satirists', 'burundi', 'habitation', 'gaelic']
# The furthest word of english -> ['kenai', 'attaining', 'broadcasting', 'solved', 'ktva']
# The nearest word of revolution  -> ['revolution', 'betrayal', 'internationale', 'neolithic', 'stalinism']
# The furthest word of revolution -> ['kenai', 'attaining', 'solved', 'broadcasting', 'ktva']
# The nearest word of against  -> ['against', 'rebelled', 'refugees', 'rushing', 'guy']
# The furthest word of against -> ['kenai', 'attaining', 'and', 'the', 'in']
# step 9500, loss 0.261
# ```
# 
# ## 取 50,000,000长度的数据
# ---
# 结果：
# ```
# step 500, loss 0.690
# step 1000, loss 0.632
# Steps -> 1000 , Similar words:
# The nearest word of english  -> ['english', 'the', 'and', 'in', 'of']
# The furthest word of english -> ['the', 'and', 'in', 'one', 'genscher']
# The nearest word of revolution  -> ['revolution', 'a', 'to', 'in', 'and']
# The furthest word of revolution -> ['in', 'and', 'the', 'one', 'genscher']
# The nearest word of against  -> ['against', 'the', 'two', 'at', 'that']
# The furthest word of against -> ['the', 'in', 'and', 'one', 'a']
# step 1500, loss 0.524
# step 2000, loss 0.411
# Steps -> 2000 , Similar words:
# The nearest word of english  -> ['english', 'for', 'most', 'since', 'on']
# The furthest word of english -> ['and', 'in', 'one', 'the', 'genscher']
# The nearest word of revolution  -> ['revolution', 'city', 'another', 'special', 'speaking']
# The furthest word of revolution -> ['and', 'one', 'in', 'the', 'genscher']
# The nearest word of against  -> ['against', 'for', 'special', 'years', 'year']
# The furthest word of against -> ['in', 'and', 'one', 'the', 'genscher']
# step 2500, loss 0.388
# step 3000, loss 0.303
# Steps -> 3000 , Similar words:
# The nearest word of english  -> ['english', 'low', 'large', 'country', 'have']
# The furthest word of english -> ['one', 'and', 'the', 'kinkel', 'genscher']
# The nearest word of revolution  -> ['revolution', 'entropy', 'tourism', 'vol', 'southwestern']
# The furthest word of revolution -> ['one', 'and', 'the', 'kinkel', 'genscher']
# The nearest word of against  -> ['against', 'city', 'special', 'only', 'year']
# The furthest word of against -> ['one', 'and', 'the', 'kinkel', 'genscher']
# step 3500, loss 0.336
# step 4000, loss 0.308
# Steps -> 4000 , Similar words:
# The nearest word of english  -> ['english', 'most', 'low', 'country', 'box']
# The furthest word of english -> ['one', 'and', 'the', 'kinkel', 'genscher']
# The nearest word of revolution  -> ['revolution', 'southwestern', 'danube', 'measurement', 'break']
# The furthest word of revolution -> ['one', 'and', 'the', 'kinkel', 'genscher']
# The nearest word of against  -> ['against', 'special', 'public', 'slavic', 'methods']
# The furthest word of against -> ['one', 'and', 'the', 'kinkel', 'genscher']
# step 4500, loss 0.271
# step 5000, loss 0.245
# Steps -> 5000 , Similar words:
# The nearest word of english  -> ['english', 'box', 'territorial', 'study', 'country']
# The furthest word of english -> ['the', 'one', 'and', 'kinkel', 'genscher']
# The nearest word of revolution  -> ['revolution', 'southwestern', 'babies', 'exempt', 'evaluated']
# The furthest word of revolution -> ['the', 'one', 'and', 'kinkel', 'genscher']
# The nearest word of against  -> ['against', 'diplomatic', 'methods', 'variables', 'city']
# The furthest word of against -> ['one', 'the', 'and', 'kinkel', 'genscher']
# step 5500, loss 0.210
# step 6000, loss 0.266
# Steps -> 6000 , Similar words:
# The nearest word of english  -> ['english', 'box', 'argue', 'coast', 'actual']
# The furthest word of english -> ['one', 'and', 'the', 'kinkel', 'genscher']
# The nearest word of revolution  -> ['revolution', 'evaluated', 'loud', 'customary', 'mandate']
# The furthest word of revolution -> ['one', 'kinkel', 'and', 'the', 'genscher']
# The nearest word of against  -> ['against', 'taxes', 'diplomatic', 'variables', 'slavic']
# The furthest word of against -> ['one', 'the', 'and', 'kinkel', 'genscher']
# step 6500, loss 0.292
# step 7000, loss 0.262
# Steps -> 7000 , Similar words:
# The nearest word of english  -> ['english', 'argue', 'health', 'participate', 'rhythm']
# The furthest word of english -> ['and', 'in', 'a', 'one', 'the']
# The nearest word of revolution  -> ['revolution', 'evaluated', 'warrant', 'dinosaurs', 'tendencies']
# The furthest word of revolution -> ['cnidocysts', 'in', 'and', 'kinkel', 'a']
# The nearest word of against  -> ['against', 'taxes', 'diplomatic', 'argue', 'served']
# The furthest word of against -> ['in', 'one', 'a', 'and', 'the']
# step 7500, loss 0.261
# step 8000, loss 0.249
# Steps -> 8000 , Similar words:
# The nearest word of english  -> ['english', 'box', 'beat', 'credits', 'rhythm']
# The furthest word of english -> ['in', 'to', 'the', 'and', 'a']
# The nearest word of revolution  -> ['revolution', 'evaluated', 'attending', 'tourism', 'organisation']
# The furthest word of revolution -> ['in', 'to', 'for', 'the', 'is']
# The nearest word of against  -> ['against', 'taxes', 'private', 'year', 'progressive']
# The furthest word of against -> ['one', 'in', 'a', 'the', 'and']
# step 8500, loss 0.284
# step 9000, loss 0.248
# Steps -> 9000 , Similar words:
# The nearest word of english  -> ['english', 'edited', 'fast', 'finally', 'box']
# The furthest word of english -> ['and', 'the', 'in', 'a', 'one']
# The nearest word of revolution  -> ['revolution', 'evaluated', 'attending', 'rolling', 'ineffective']
# The furthest word of revolution -> ['cnidocysts', 'in', 'kinkel', 'with', 'to']
# The nearest word of against  -> ['against', 'taxes', 'need', 'spirituality', 'driving']
# The furthest word of against -> ['one', 'in', 'a', 'the', 'and']
# step 9500, loss 0.238
# step 10000, loss 0.281
# Steps -> 10000 , Similar words:
# The nearest word of english  -> ['english', 'entertainers', 'korty', 'charted', 'edited']
# The furthest word of english -> ['in', 'to', 'cnidocysts', 'and', 'a']
# The nearest word of revolution  -> ['revolution', 'attending', 'concentrating', 'evaluated', 'jewelry']
# The furthest word of revolution -> ['cnidocysts', 'kinkel', 'in', 'a', 'and']
# The nearest word of against  -> ['against', 'taxes', 'need', 'spirituality', 'chalcogen']
# The furthest word of against -> ['a', 'in', 'one', 'cnidocysts', 'and']
# ```
# 

# In[1]:


import io
import os
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph.nn import Embedding


# 下载语料用来训练word2vec
def download():
    # 可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    # 使用python的requests包下载数据集到本地
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    # 把下载后的文件存储在当前目录的text8.txt文件内
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()

download()

# 读取text8数据
def load_text8():
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus


corpus = load_text8()
corpus = corpus[:50000000]

# 打印前500个字符，简要看一下这个语料的样子
print(corpus[:500])


# 对语料进行预处理（分词）
def data_preprocess(corpus):
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")

    return corpus


corpus = data_preprocess(corpus)
print(corpus[:50])


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    # 一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    # 构造3个不同的词典，分别存储，
    # 每个词到id的映射关系：word2id_dict
    # 每个id出现的频率：word2id_freq
    # 每个id到词典映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus


corpus = convert_corpus_to_id(corpus, word2id_dict)
print("%d tokens in the corpus" % len(corpus))
print(corpus[:50])


# 使用二次采样算法（subsampling）处理语料，强化训练效果
def subsampling(corpus, word2id_freq):
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus


corpus = subsampling(corpus, word2id_freq)
print("%d tokens in the corpus" % len(corpus))
print(corpus[:50])


# 构造数据，准备模型训练
# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
# negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。
def build_data(corpus, word2id_dict, word2id_freq, max_window_size=3, negative_sample_num=4):
    # 使用一个list存储处理好的数据
    dataset = []

    # 从左到右，开始枚举每个中心点的位置
    for center_word_idx in range(len(corpus)):
        # 以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        # 当前的中心词就是center_word_idx所指向的词
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看成是正样本
        positive_word_range = (
        max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1] + 1) if
                                    idx != center_word_idx]

        # 对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            # 首先把（中心词，正样本，label=1）的三元组数据放入dataset中，
            # 这里label=1表示这个样本是个正样本
            dataset.append((center_word, positive_word, 1))

            # 开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate not in positive_word_candidates:
                    # 把（中心词，正样本，label=0）的三元组数据放入dataset中，
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((center_word, negative_word_candidate, 0))
                    i += 1

    return dataset


dataset = build_data(corpus, word2id_dict, word2id_freq)
for _, (center_word, target_word, label) in zip(range(50), dataset):
    print("center_word %s, target %s, label %d" % (id2word_dict[center_word],
                                                   id2word_dict[target_word], label))


# 构造mini-batch，准备对模型进行训练
# 我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
# 并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):
    # center_word_batch缓存batch_size个中心词
    center_word_batch = []
    # target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)

        for center_word, target_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype("int64"),                       np.array(target_word_batch).astype("int64"),                       np.array(label_batch).astype("float32")
                center_word_batch = []
                target_word_batch = []
                label_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"),               np.array(target_word_batch).astype("int64"),               np.array(label_batch).astype("float32")


for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)


# 定义skip-gram训练网络结构
# 这里我们使用的是paddlepaddle的1.6.1版本
# 一般来说，在使用fluid训练的时候，我们需要通过一个类来定义网络结构，这个类继承了fluid.dygraph.Layer
class SkipGram(fluid.dygraph.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        # vocab_size定义了这个skipgram这个模型的词表大小
        # embedding_size定义了词向量的维度是多少
        # init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # 使用paddle.fluid.dygraph提供的Embedding函数，构造一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的名称为：embedding_para
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / embedding_size, high=0.5 / embedding_size)))

        # 使用paddle.fluid.dygraph提供的Embedding函数，构造另外一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的名称为：embedding_para
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        # 跟上面不同的是，这个参数的名称跟上面不同，因此，
        # embedding_out_para和embedding_para虽然有相同的shape，但是权重不共享
        self.embedding_out = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / embedding_size, high=0.5 / embedding_size)))

    # 定义网络的前向计算逻辑
    # center_words是一个tensor（mini-batch），表示中心词
    # target_words是一个tensor（mini-batch），表示目标词
    # label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    # 用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
    def forward(self, center_words, target_words, label):
        # 首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
        # 这里center_words和eval_words_emb查询的是一个相同的参数
        # 而target_words_emb查询的是另一个参数
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # center_words_emb = [batch_size, embedding_size]
        # target_words_emb = [batch_size, embedding_size]
        # 我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = fluid.layers.elementwise_mul(center_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim=-1)
        word_sim = fluid.layers.reshape(word_sim, shape=[-1])
        pred = fluid.layers.sigmoid(word_sim)

        # 通过估计的输出概率定义损失函数，注意我们使用的是sigmoid_cross_entropy_with_logits函数，将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)

        # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss


batch_size = 512
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001


# 定义一个使用word-embedding查询同义词的函数
# 这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
# 我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
# 具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding，两者计算Cos得出所有词对查询词的相似度得分向量，排序取top k放入indices列表
def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices_pos = np.argpartition(flat, -k)[-k:]
    indices_neg = np.argpartition(flat, k)[-k:]
    indices_pos = indices_pos[np.argsort(-flat[indices_pos])]
    indices_neg = indices_neg[np.argsort(-flat[indices_neg])]
    tmp_pos = []
    tmp_neg = []
    for i in indices_pos:
        tmp_pos.append(str(id2word_dict[i]))
    for i in indices_neg:
        tmp_neg.append(str(id2word_dict[i]))
    print('The nearest word of %s  -> %s' % (query_token, str(tmp_pos)))
    print('The furthest word of %s -> %s' % (query_token, str(tmp_neg)))


# 将模型放到GPU上训练（fluid.CUDAPlace(0)），如果需要指定CPU，则需要改为fluid.CPUPlace()
with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    # 通过我们定义的SkipGram类，来构造一个Skip-gram模型网络
    skip_gram_model = SkipGram(vocab_size, embedding_size)
    # 构造训练这个网络的优化器
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list=skip_gram_model.parameters())

    # 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
    for center_words, target_words, label in build_batch(
            dataset, batch_size, epoch_num):
        # 使用fluid.dygraph.to_variable函数，将一个numpy的tensor，转换为飞桨可计算的tensor
        center_words_var = fluid.dygraph.to_variable(center_words)
        target_words_var = fluid.dygraph.to_variable(target_words)
        label_var = fluid.dygraph.to_variable(label)

        # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
        pred, loss = skip_gram_model(
            center_words_var, target_words_var, label_var)

        # 通过backward函数，让程序自动完成反向计算
        loss.backward()
        # 通过minimize函数，让程序根据loss，完成一步对参数的优化更新
        adam.minimize(loss)
        # 使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
        skip_gram_model.clear_gradients()

        # 每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        step += 1
        if step % 500 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        # 经过10000个mini-batch，打印一次模型对eval_words中的10个词计算的同义词
        # 这里我们使用词和词之间的向量点积作为衡量相似度的方法
        # 我们只打印了5个最相似的词
        if step % 1000 == 0:
            print('Steps -> %s , Similar words:' % step)
            get_similar_tokens('english', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('revolution', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('against', 5, skip_gram_model.embedding.weight)

        if step == 50000:
            exit(0)

