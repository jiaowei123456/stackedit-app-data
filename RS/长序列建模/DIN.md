# Deep Interest Network for Click-Through Rate Prediction
[原文链接](https://doi.org/10.1145/3383313.3412236)
## 0 摘要：
点击率预测在工业应用中是一项至关重要的任务，例如在线广告领域。最近，基于深度学习的模型已被提出，它们遵循一种类似的“嵌入&多层感知机”模式。在这些方法中，首先会将大规模稀疏输入特征映射为低维嵌入向量，然后以分组的方式将其转换为固定长度的向量，最后将它们连接在一起，输入到多层感知机（MLP）中以学习特征之间的非线性关系。这样，无论选择何种候选广告，用户特征都会被压缩为一个固定长度的表示向量。固定长度向量的使用会成为瓶颈，这给“嵌入&多层感知机”方法从丰富的历史行为中有效地捕捉用户多样化的兴趣带来了困难。在本文中，我们提出了一种新颖的模型：深度兴趣网络（DIN），它通过设计一个局部激活单元来根据特定广告从历史行为中自适应地学习用户兴趣的表示。这个表示向量在不同的广告中会有所不同，极大地提高了模型的表达能力。此外，我们开发了两种技术：小批量感知正则化和数据自适应激活函数，它们能够帮助训练具有数亿参数的工业深度网络。在两个公开数据集以及一个拥有超过 20 亿样本的阿里巴巴实际生产数据集上的实验表明，所提出的方法是有效的，其性能优于当前最先进的方法。DIN 现已成功部署在阿里巴巴的在线展示广告系统中，为主要流量提供服务。
## 1 论文解决的问题：
* 在嵌入式和多层感知器方法中，具有有限维度的用户表示向量会成为表达用户多样化兴趣的瓶颈。
* 为了使表示能够充分地表达用户的各种兴趣，固定长度向量的维度需要大幅扩展。不幸的是，这会极大地增加学习参数的规模，并在有限的数据下加剧过拟合的风险。此外，这还会增加计算和存储方面的负担，这对于一个工业化的在线系统来说是难以承受的。
* 在预测候选广告时，没有必要将某一用户的所有不同兴趣都压缩成同一个向量，因为只有用户的一部分兴趣会影响其行为
## 2 论文创新点：
* 通过考虑给定候选广告的历史行为的相关性，自适应地计算用户兴趣的表示向量。DIN在生成用户embedding vector的时候加入了一个activation unit层，这一层产生了每个用户行为的权重。
* 通过引入局部激活单元，DIN 通过软搜索历史行为的相关部分来关注相关的用户兴趣，并采用加权求和池化来获得与候选广告相关的用户兴趣的表示。与候选广告相关性更高的行为会获得更高的激活权重，并主导用户兴趣的表示。
* 我们设计了一种数据自适应激活函数，它扩展了常用的 PReLU[12]，通过根据输入的分布自适应地调整修正点，且已被证明有助于训练具有稀疏特征的工业网络。
## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-09/CG0JH6ExEsL5wPKH.png)

### 用户emb权重注意力机制代码
传统的Attention机制中，给定两个item embedding，比如u和v，通常是直接做点积uv或者uWv，其中W是一个|u|x|v|的权重矩阵，但这篇paper中阿里显然做了更进一步的改进，着重看上图右上角的activation unit，**首先是把u和v以及u v的element wise差值向量合并起来作为输入，然后喂给全连接层，最后得出权重**，这样的方法显然损失的信息更少。

代码输入queries, keys, keys_length，将q，k经过拼接之后直接输入DNN输出维度与key的seq长度相同，获得由queries加权后的keys，或者权重系数（经过softmax），然后输出加权后的key
特点：1、用于注意力计算的值不仅仅是queries, keys，还包括queries - keys, queries * keys。
2、keys_length映射了keys维度[B, T, X]中的T维度的有效位数，用于掩码，去掉冗余信息
```Python
def attention(queries, keys, keys_length,  
              ffn_hidden_units=[80, 40], ffn_activation=dice,  
              queries_ffn=False, queries_activation=prelu,  
              return_attention_score=False):  
    """  
  
    :param queries: [B, H]    :param keys: [B, T, X]    :param keys_length: [B]    :param queries_ffn: 是否对queries进行一次ffn  
    :param queries_activation: queries ffn的激活函数  
    :param ffn_hidden_units: 隐藏层的维度大小  
    :param ffn_activation: 隐藏层的激活函数  
    :param return_attention_score: 是否返回注意力得分  
    :return: attention_score=[B, 1, T] or attention_outputs=[B, H]  
    """    if queries_ffn:  
        queries = tf.layers.dense(queries, keys.get_shape().as_list()[-1], name='queries_ffn')  
        queries = queries_activation(queries)  
    queries_hidden_units = queries.get_shape().as_list()[-1]  
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  
    hidden_layer = dnn_layer(din_all, ffn_hidden_units, ffn_activation, use_bn=False, scope='attention')  
    outputs = tf.layers.dense(hidden_layer, 1, activation=None)  
    outputs = tf.reshape(outputs, [-1, 1, tf.shape(keys)[1]])  # [B, 1, T] 
    # Mask  
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]  # 根据 keys_length（每个样本的有效时间步数）创建一个布尔掩码张量。
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]  
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]  
  
    # Scale    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)  
  
    # Activation  
    attention_score = tf.nn.softmax(outputs)  # [B, 1, T]  
  
    if return_attention_score:  
        return attention_score  
  
    # Weighted sum  
    attention_outputs = tf.matmul(attention_score, keys)  # [B, 1, H]  
  
    return tf.squeeze(attention_outputs)
```

## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbNjk0OTYzNzc5XX0=
-->