# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender
[原文链接]([arxiv.org/pdf/2510.26104](https://arxiv.org/pdf/2510.26104))
## 0 摘要：
在推荐系统中，扩展特征交叉模块（例如：Wukong，Rankmixer)或用户行为序列模块（Longer）。取得了显著的成功。然而,这些努力通常是在不同的轨道上进行的,这不仅阻碍了双向信息交换,还阻碍了统一的优化和扩展。在本文中,我们提出了一个统一的变压器骨干,同时执行用户行为序列建模和特征交互。OneTrans采用统一的tokenizer将顺序和非顺序属性转换为单个令牌序列。在向非顺序令牌分配特定于特定于特定参数的参数时,堆叠的OneTrans块将共享参数,在类似的顺序令牌上共享参数。通过因果关系和交叉请求KV缓存,OneTrans可以预先计算和缓存中间表示,在训练和推理期间显著地降低了计算成本。在工业水平数据集上的实验结果表明,通过增加的参数有效地提高了一阶式的尺度,持续优于较强的基线,并在在线a / B测试中以每个用户GMV的速度提高5.68%的升力。

## 背景


## 1 论文解决的问题：


## 2 论文创新点：


### 2.1 预训练数据集的构建：


### 2.1 技巧：


## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAwNTI5MTc5NF19
-->