# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender
[原文链接]([arxiv.org/pdf/2510.26104](https://arxiv.org/pdf/2510.26104))
## 0 摘要：
在推荐系统中，扩展特征交叉模块（例如：Wukong，Rankmixer）或用户行为序列模块（Longer）。取得了显著的成功。然而，这些努力通常是在不同的方向上进行的，这阻碍了双向信息交换以及统一的优化和扩展。在本文中，我们提出了一个统一的Transformer架构，同时执行用户行为序列建模和特征交叉。OneTrans采用统一的tokenizer将序列和非序列特征转换为单个Token序列。堆叠的OneTrans Block中，非序列Token分配特定的参数，序列Token使用共有参数。通过因果注意力和交叉请求KV缓存，OneTrans可以预计算和缓存中间表示，在训练和推理期间显著地降低了计算成本。在工业水平数据集上的实验结果表明，通过参数增加方式有效地展示了scaling law，持续优于较强的Baseline，在线A/B测试中per u GMV展现出5.68%的提升。

## 背景
1. 主流精排模型包含两个重要模块：（1）序列建模：利用局部注意力或 Transformer 编码器将用户的多行为序列编码为具有候选意识的表示形式。(DIN，SIM，ETA，TWIN，TWINv2，LONGER)（2）特征交叉：通过FM（DeepFM）或显式交叉网络（DCN，DCNv2，SENet，PEPNet）或特征组上的注意力机制（RankMixer）来学习非序列特征之间的高阶交叉。（先序列建模，将长序列压缩为短特征，过特征交叉获得高阶表示，然后过下游任务）
2. 大模型表现出scaling law，推荐系统也想要（LONGER， WuKong， RankMixer）

## 1 论文解决的问题：


## 2 论文创新点：


### 2.1 预训练数据集的构建：


### 2.1 技巧：


## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4MTI4MTIzMjhdfQ==
-->