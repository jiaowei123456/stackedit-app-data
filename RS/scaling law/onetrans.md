# OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender
[原文链接]([arxiv.org/pdf/2510.26104](https://arxiv.org/pdf/2510.26104))
## 0 摘要：
在推荐系统中，扩展特征交叉模块（例如：Wukong，Rankmixer）或用户行为序列模块（Longer）。取得了显著的成功。然而，这些努力通常是在不同的方向上进行的，这阻碍了双向信息交换以及统一的优化和扩展。在本文中，我们提出了一个统一的Transformer架构，同时执行用户行为序列建模和特征交叉。OneTrans采用统一的tokenizer将序列和非序列特征转换为单个Token序列。堆叠的OneTrans Block中，非序列Token分配特定的参数，序列Token使用共有参数。通过因果注意力和交叉请求KV缓存，OneTrans可以预计算和缓存中间表示，在训练和推理期间显著地降低了计算成本。在工业水平数据集上的实验结果表明，通过参数增加方式有效地展示了scaling law，持续优于较强的Baseline，在线A/B测试中per u GMV展现出5.68%的提升。

## 背景
1. 主流精排模型包含两个重要模块：（1）序列建模：利用局部注意力或 Transformer 编码器将用户的多行为序列编码为具有候选意识的表示形式。(DIN，SIM，ETA，TWIN，TWINv2，LONGER)（2）特征交叉：通过FM（DeepFM）或显式交叉网络（DCN，DCNv2，SENet，PEPNet）或特征组上的注意力机制（RankMixer）来学习非序列特征之间的高阶交叉。（先序列建模，将长序列压缩为短特征，过特征交叉获得高阶表示，然后过下游任务）
2. 大模型表现出scaling law，推荐系统也想要（LONGER， WuKong， RankMixer）

## 1 论文解决的问题：
1. 序列建模然后特征交叉的pipline限制了双向信息流动，限制了静态/上下文特征构建序列表征。
2. 两个模块分离会串行执行并增加延迟，而统一的Transformer式Backbone可以重复使用 LLM 的优化，例如KV缓存、内存高效注意力和混合精度等，以实现更有效的扩展。

## 2 论文创新点：
1. 在本文中，提出了 ONETRANS 这一创新的架构模式，它具有统一的 Transformer 基础架构，能够同时进行用户行为序列建模和特征交互。
2. ONETRANS的基础架构实现了双向信息流动。它采用了一个统一的tokenizer，将序列特征和非序列特征都转换为一个单一的Token或者Token序列，然后由一系列堆叠的 ONETRAN 块（一种专为工业推荐系统设计的 Transformer 变体）进行处理。为了适应推荐系统中各种不同的标记来源，与语言模型中仅包含文本标记的情况不同，每个 ONETRAN 块采用了类似于 HiFormer [11] 的混合参数化方式。具体来说，所有序列标记（来自序列特征）共享一组 Q/K/V 和 FFN 权重，而每个非序列标记（来自非序列特征）则接收标记特定的参数以保留其独特的语义。

### 2.1 预训练数据集的构建：


### 2.1 技巧：


## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTExMDg1NDE4NV19
-->