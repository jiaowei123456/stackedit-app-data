# RankMixer: Scaling Up Ranking Models in Industrial Recommenders
[原文链接](https://www.arxiv.org/pdf/2507.15551)
## 0 摘要：
近期在大型语言模型（LLM）方面取得的进展激发了人们对扩大推荐系统的兴趣，但仍有两个实际障碍。首先，工业推荐系统的训练和提供服务的成本必须符合严格的**延迟限制和高每秒查询量（QPS）需求**。其次，排名模型中大多数由人类设计的**特征交叉模块是从 CPU 时代继承下来的**，无法充分利用现代 GPU，导致模型浮点运算利用率（MFU）低且可扩展性差。我们引入了 RankMixer，这是一种面向统一且可扩展的特征交互架构的硬件感知模型设计。RankMixer 保留了 Transformer 的高并行性，同时用**多头token混合模块**取代了**二次自注意力机制**，以提高效率。此外，RankMixer 通过每个标记的前馈网络（Per-token FFNs）同时保持了对不同特征子空间的建模和跨特征空间的交互。我们进一步将其扩展到十亿参数，并采用**稀疏混合专家（Sparse-MoE）变体**以提高投资回报率（ROI）。采用**动态路由策略**来解决专家训练的不足和不平衡问题。实验表明，RankMixer 在万亿规模的生产数据集上具有出色的扩展能力。通过用 RankMixer 替换之前多样化的手工低模型浮点利用率（MFU）模块，我们将模型的 MFU 从 4.5% 提升至 45%，并将我们的在线排序模型参数规模扩大了两个数量级，同时保持了大致相同的推理延迟。我们通过在两个核心应用场景（推荐和广告）中进行在线 A/B 测试，验证了 RankMixer 的通用性。最后，我们在不增加服务成本的情况下，将 10 亿密集参数的 RankMixer 推广到全流量服务，使用户活跃天数提高了 0.3%，总应用内使用时长提高了 1.08%。

## 背景
1、简单的堆叠特征交互层，结构未修改，效果微弱甚至为负面。
2、DHEN和Wukong设计创新的深度神经网络结构以提高扩展性能

## 1 论文解决的问题：
1、必须严格遵守严格的延迟限制，并支持极高的每秒查询数（QPS）。
2、原始rank模型的注意力机制主要在CPU计算阶段提出，核心操作大多受内存限制而非计算限制，在现代 GPU 上，这导致了较差的 GPU 并行性以及极低的 MFU（模型运算次数利用率），
3、架构应与硬件相匹配，以在现代 GPU 上实现最大化的多功能单元（MFU）和计算吞吐量。
4、模型设计必须利用推荐数据的特性，例如异构特征空间以及数百个字段之间的个性化跨特征交互。

## 2 论文创新点：
1. Multi-head token mixing ：只通过无参数操作符获得跨token特征交互。该策略在性能和计算效率方面优于自注意机制。
2. Per-token feed-forward networks (FFNs)：通过为不同的特征子空间建模分配独立的参数，极大地扩展了模型容量，解决了特征空间间的控制问题。
3. Sparse Mixture-of-Experts (MoE)：通过针对不同的数据动态激活每个标记的特定子集专家，我们能够以最小的计算成本显著提高模型的容量。


## 3 模型结构与实现代码：
### 3.1 整体框架：
![输入图片说明](/imgs/2025-12-13/iXt3rIjZqdbgMP4S.png)

输入为T个token，经过连续L个Rankmixer以及平均池化后输出，每个 RankMixer 块有两个主要组成部分：（1）多头token mixing，以及（2）每个token的per-token FFN（PFFN）层，如图 1 所示。

$S_{n-1} = \operatorname{LN}\!\left( \operatorname{TokenMixing}(X_{n-1}) + X_{n-1} \right)$
$X_n = \mathrm{LN} \left( \mathrm{PFFN} \left( S_{n-1} \right) + S_{n-1} \right)$
### 3.2 输入层和特征token化：
1. 用户特征：包括用户 ID 及其他用户信息等
2. 物品特征：视频 ID、作者 ID 等
3. 序列特征：通过序列模块处理后的序列特征用于捕捉时间相关性
4. 交叉特征：得到嵌入向量和交叉特征

Tokenization：为了实现高效的并行计算，不同维度的embedding必须转换为维度对齐的向量，这些向量被称为特征Tokenization。


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE0MDYwMDgwNiwtNzYxOTEzOTg5LDY0Mj
U1ODcyOSwxODY2MDA2ODE1LDIwNDkxMzg3MDUsLTg2NTE5MzM1
M119
-->