# Deep Interest Network for Click-Through Rate Prediction
[原文链接](https://doi.org/10.1145/3383313.3412236)
## 0 摘要：
点击率预测在工业应用中是一项至关重要的任务，例如在线广告领域。最近，基于深度学习的模型已被提出，它们遵循一种类似的“嵌入&多层感知机”模式。在这些方法中，首先会将大规模稀疏输入特征映射为低维嵌入向量，然后以分组的方式将其转换为固定长度的向量，最后将它们连接在一起，输入到多层感知机（MLP）中以学习特征之间的非线性关系。这样，无论选择何种候选广告，用户特征都会被压缩为一个固定长度的表示向量。固定长度向量的使用会成为瓶颈，这给“嵌入&多层感知机”方法从丰富的历史行为中有效地捕捉用户多样化的兴趣带来了困难。在本文中，我们提出了一种新颖的模型：深度兴趣网络（DIN），它通过设计一个局部激活单元来根据特定广告从历史行为中自适应地学习用户兴趣的表示。这个表示向量在不同的广告中会有所不同，极大地提高了模型的表达能力。此外，我们开发了两种技术：小批量感知正则化和数据自适应激活函数，它们能够帮助训练具有数亿参数的工业深度网络。在两个公开数据集以及一个拥有超过 20 亿样本的阿里巴巴实际生产数据集上的实验表明，所提出的方法是有效的，其性能优于当前最先进的方法。DIN 现已成功部署在阿里巴巴的在线展示广告系统中，为主要流量提供服务。
## 1 论文解决的问题：
* 在嵌入式和多层感知器方法中，具有有限维度的用户表示向量会成为表达用户多样化兴趣的瓶颈。
* 为了使表示能够充分地表达用户的各种兴趣，固定长度向量的维度需要大幅扩展。不幸的是，这会极大地增加学习参数的规模，并在有限的数据下加剧过拟合的风险。此外，这还会增加计算和存储方面的负担，这对于一个工业化的在线系统来说是难以承受的。
* 在预测候选广告时，没有必要将某一用户的所有不同兴趣都压缩成同一个向量，因为只有用户的一部分兴趣会影响其行为
## 2 论文创新点：
* 通过考虑给定候选广告的历史行为的相关性，自适应地计算用户兴趣的表示向量。
* 通过引入局部激活单元，DIN 通过软搜索历史行为的相关部分来关注相关的用户兴趣，并采用加权求和池化来获得与候选广告相关的用户兴趣的表示。与候选广告相关性更高的行为会获得更高的激活权重，并主导用户兴趣的表示。
## 3 相关工作：
* 直接的多塔线性组合，学习线性组合的权重；融合来自不同任务的表示。a，f
* MOE 通过门控网络将专家进行组合；MMOE 针对每个任务使用不同的门控机制；MRAN采用多头自注意力机制来在不同的特征集上学习不同的表示子空间。h
* 
![输入图片说明](/imgs/2025-07-08/4sKGptX6jkr7uNdd.png)
## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-08/gDJqongZ5GfUFQnO.png)
### 整体代码
```Python
task_fea = [emb for i in range(self.task_num + 1)] # task1 input ,task2 input,..taskn input, share_expert input  
for i in range(self.layers_num):  
    share_output=[expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]  
    task_output_list=[]  
    for j in range(self.task_num):  
        task_output=[expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]  
        task_output_list.extend(task_output)  
        mix_ouput=torch.cat(task_output+share_output,dim=1)  
        gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)  
        task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)  
    if i != self.layers_num-1:#最后一层不需要计算share expert 的输出  
        gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)  
        mix_ouput = torch.cat(task_output_list + share_output, dim=1)  
        task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)  
  
results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
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
eyJoaXN0b3J5IjpbMjkwMzYyOTc3LDE1NDg1NTMxMTZdfQ==
-->