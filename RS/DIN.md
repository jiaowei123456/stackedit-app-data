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
```Python
def attention(queries, keys, keys_length):  
    '''  
        queries:     [B, H]    [batch_size,embedding_size]        keys:        [B, T, H]   [batch_size,T,embedding_size]        keys_length: [B]        [batch_size]        #T为历史行为序列长度  
    '''  
    # (?,32)->(None,32)->32  
    # tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变  
    # tf.shape(keys)[1]==T  
    # 对queries的维度进行reshape  
    # (?,T,32)这里是为了让queries和keys的维度相同而做的操作  
    # (?,T,128)把u和v以及u v的element wise差值向量合并起来作为输入，  
    # 然后喂给全连接层，最后得出两个item embedding，比如u和v的权重，即g(Vi,Va)  
  
    queries_hidden_units = queries.get_shape().as_list()[-1]  
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B*T*4H  
  
    # 三层全链接(d_layer_3_all为训练出来的atteneion权重）  
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')  
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')  
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1  
  
    # 为了让outputs维度和keys的维度一致  
    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T  
  
    #  bool类型 tf.shape(keys)[1]为历史行为序列的最大长度，keys_length为人为设定的参数，  
    #  如tf.sequence_mask(5，3)  即为array[True,True,True,False,False]  
    #  函数的作用是为了后面补齐行为序列，获取等长的行为序列做铺垫  
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  
  
    # 在第二维增加一维，也就是由B*T变成B*1*T  
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T  
  
    # tf.ones_like新建一个与output类型大小一致的tensor，设置填充值为一个很小的值，而不是0,padding的mask后补一个很小的负数，这样softmax之后就会接近0  
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  
  
    # 填充，获取等长的行为序列  
    # tf.where(condition， x, y),condition是bool型值，True/False，返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素  
    # 由于是替换，返回值的维度，和condition，x ， y都是相等的。  
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T  
  
    # Scale（缩放）  
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)  
    # Activation  
    outputs = tf.nn.softmax(outputs)  # B * 1 * T  
    # Weighted Sum outputs=g(Vi,Va)   keys=Vi    # 这步为公式中的g(Vi*Va)*Vi  
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))  
    return outputs
```
### 4.1 专家网络
可以理解为每一层都有很多专家，专家可以分为专用专家(num_task)和通用专家(1)，每一个专家有自己的子全连接-专家个数。
#### 共享专家代码实现
```Python
share_output=[expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]] # 输入为（batch_size, input_dim），share_experts为layers_num层，每一层有shared_expert_num个全连接层——MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)，最后输出为（batch_size, 1, bottom_mlp_dims[i]），有shared_expert_num个
```
#### 特殊专家代码实现
```Python
task_output=[expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]] # 输入为（batch_size, input_dim），task_experts为layers_num层，每一层有specific_expert_num个全连接层——MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)，最后输出为（batch_size, 1, bottom_mlp_dims[i]），有specific_expert_num个。注：特殊专家网络mlp数量为layers_num*task_num*specific_expert_num
```
#### 特殊专家门控代码实现
```Python
gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1) # 每一个任务都有一个对应的门控结果，因此门控网络数量为layers_num*task_num，每一个网络为：torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1))，因此输出为（batch_size, 1, shared_expert_num + specific_expert_num）
```
#### 特殊专家加权输出
```Python
mix_ouput = torch.cat(task_output + share_output,dim=1)   #shared_expert_num个共享专家，specific_expert_num个特殊专家拼接
task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1) # 加权输出，输出维度为（batch_size, 1, bottom_mlp_dims[i]）
```
### 4.2 中间层的混合专家加权输出（非最后一层）
```Python
if i != self.layers_num-1:#最后一层不需要计算share expert 的输出  
    gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)  #（batch_size, 1, shared_expert_num + specific_expert_num*task_num）
    mix_ouput = torch.cat(task_output_list + share_output, dim=1)   #（batch_size, shared_expert_num + specific_expert_num*task_num，bottom_mlp_dims[i]）
    task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)  #（batch_size,1，bottom_mlp_dims[i]）
```
### 4.3 最终每个任务加一个全连接层预测最终输出
```Python
results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)] #使用sigmoid作为激活函数。输出（batchsize，num_task）
```

## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNTU0ODUwNjksMTM5NzE3NDY1MSw0Mj
M2OTM3ODcsNTQ1NzI4NTgxLDE1NDg1NTMxMTZdfQ==
-->