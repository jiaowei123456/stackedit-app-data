# LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders
[原文链接]([LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/pdf/2505.04421))

## 0 摘要：
对超长用户行为序列进行建模对于确定工业推荐修复系统的长期和短期偏好至关重要。现有的解决方案通常依赖于两阶段检索或间接建模范式，导致上下游不一致和计算效率低下。在本文中，我们提出了一个用于gpu高效推荐器的长序列优化transformer结构**LONGER**。LONGER包含了(i)一个全局token机制，用于在长上下文内稳定注意力；（ii）一个带有轻量级inner transformer和混合注意策略的token合并模块，以降低二次复杂度；（iii）进行了一系列工程优化，包括混合精度训练和激活重计算、KV 缓存服务，以及用于统一基于 GPU 的密集和稀疏参数更新的全同步模型训练和服务框架。在字节跳动的广告和电商服务中，LONGER 在离线指标和在线 A/B 测试中始终优于强大的基线模型，验证了其持续的有效性和工业级scaling laws。目前，LONGER 已在字节跳动的数十个具有影响力的现实场景中得到验证和全面部署，为数十亿用户提供服务。

## 背景
1. 全面建模长序列（长度＞$10^3$）对于推荐的准确性和多样性具有显著优势，并有助于缓解信息茧房现象。
2. Two-stage retrieval、Pre-trained User Embeddings和Memory-augmented Models等策略显著提高了计算效率，但它们不可避免地会牺牲原始的完整序列信息，原因在于上下游信息的不一致或者对原始超长序列的间接感知，因此这些方法实际上只是朝着端到端长序列建模这一目标所经历的中间阶段。


## 1 论文解决的问题：
1. 推进超长序列的端到端建模，以及不断扩展序列长度和完善长序列建模的体系结构，是下一代序列建模框架的关键要求。


## 2 论文创新点：
1. 一种用于gpu高效推荐的长序列优化transformer结构。它通过优化transformer结构，以端到端的方式将用户序列建模长度扩展到10,000。
2. LONGER通过token合并和混合注意力策略充分提高了计算效率，减少了约50%的FLOPs，并被证明在性能上几乎是无损的。


## 3 模型结构：
### 3.1 Problem Statement：
推荐系统常用数学建模，跳过，后续使用到的变量会单独介绍。

### 3.2 Overall Framework：
![输入图片说明](/imgs/2025-12-16/t0HtAyrPKOUi68TF.png)
上图展示了本文提出的模型Longer的整体架构。该框架集成了全局token、token合并、混合注意机制和训练服务优化，以实现高效和可扩展的长序列建模。

### 3.3 Global Tokens
![输入图片说明](/imgs/2025-12-17/bcbBzWDQy2kWs2yi.png)
本文引入了“全局token”作为附加到输入序列中的辅助表示，主要有两个作用：
1. 全局token充当集中式信息锚点，增强了用户历史记录、上下文属性和候选项目之间的特征交互。
2. 稳定了长序列中的注意力动态，特别是在稀疏注意力配置下。引入少量全局token可缓解“注意力汇聚”效应，即深层注意力层过度关注早期token。
这些token作为锚点，保持注意力的多样性，并保留长程依赖关系建模。

### 3.4 Token Merge
![输入图片说明](/imgs/2025-12-17/2yOQDN5NLTSm0zX4.png)
原始transformer的二次注意力计算复杂度为$O(L^2d)$，其中L为序列长度，d为embedding维度，本文使用Token Merge策略避免丢失远程依赖关系：将相邻的K个token分组并压缩为更短的序列（相邻指的是时间戳相邻？），K个token内部会过InnerTrans（transformer），通过缩短序列长度来降低计算复杂度，同时增加参数量。

### 3.5 LONGER Model Structure
#### 3.5.1 Input Generation
输入包括序列token以及全局token。

同时输入端采用了两种形式的位置编码：
1. 将量化每次用户交互与目标项目之间时间距离的绝对时差特征作为sideinfo并连接到每个项目emb中；
2. 可学习的绝对位置嵌入：对添加到项目emb的序列中的每个token的位置进行编码。

在位置编码之后，token通过MLP来获得它们的输入（这个MLP的输入输出维度相同吗？）：
$\mathbf{R} \in \mathbb{R}^{(m+L) \times d} = [\mathbf{G} \in \mathbb{R}^{m \times d}, \mathbf{H} \in \mathbb{R}^{L \times d}]$
其中G表示全局token长度为m，H表示序列token长度为L。

然后获取查询矩阵，其中G为全部的全局token，Hs为抽样获得的k个序列token，注意这个k和3.4的K不同，抽样的策略效果后面有做实验进行比较。
$\mathbf{O} = [\mathbf{G}; \mathbf{H}_s]$

#### 3.5.2 Cross-Causal Attention (First Layer)
![输入图片说明](/imgs/2025-12-17/KHWNHKaaay6lGRd2.png)
第一层是交叉因果注意力计算：
$Q = O W_Q, \quad K = R W_K, \quad V = R W_V$

$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{M} \right) \mathbf{V}$
其中O是目标query，R是key，value。M是一个右上值全为0左下值全为-inf的三角矩阵。
$\mathbf{M}_{i,j} = \begin{cases} 0, & \text{if } j \geq i, \text{ where } \{i, j\} \in [1, m + L] \\ -\infty, & \text{otherwise} \end{cases}$
因果掩码M设计一方面保持了序列中各元素之间的时间相关性。另一方面，它确保了序列对候选元素的不可见性，从而实现了 KV 缓存服务机制。在计算完注意力值后，结果会通过前馈网络（FFN）进行进一步处理。

#### 3.5.3 Self-Causal Attention (Subsequent Layers).
![输入图片说明](/imgs/2025-12-17/kRO2ZaF8Pi7GY9QQ.png)
在交叉因果注意力层之后，后续的层由N个自因果注意力层组成。这些层专注于学习采样token序列内的内部关系。每个自因果注意力层之后都接有一个FFN。自因果注意力机制的计算使用了类似的公式：
$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{M} \right) \mathbf{V}$
其中Q, K, V全为上一层的输出经过一个线性层。

### 3.6 Training and Deployment Optimization
#### 3.6.1 Training Framework
训练流程始于以批处理或流式形式进行的数据摄取，随后通过 Fountain 模块进行预处理。处理后的训练数据随后被分发到多个 GPU 运行器中。该框架的一个关键特性是其统一的参数存储和训练架构。无论是密集参数还是稀疏参数，都可在 GPU 机器上同步进行存储和更新。为了更好地适应推荐系统中的特征分布模式，该框架采用了分层存储系统来处理稀疏嵌入，能够高效支持大型嵌入表。高频特征存储在高带宽的 GPU 内存（HBM）中，中频特征存放在 CPU 主内存（MEM）中，低频特征则被卸载到本地固态硬盘（SSD）上。
#### 3.6.2 Mixed Precision Training and Recompute
为了在训练过程中缓解 GPU 内存压力，采用了重计算策略并结合了混合精度训练。对于梯度计算，使用反向模式自动微分，这种方式比正向模式更高效，但需要存储前向传播过程中的所有中间激活值。这些激活值可能会成为内存的主要瓶颈。为了解决这个问题，允许在前向传播过程中丢弃选定的激活值，并在反向传播过程中重新计算它们。这是以牺牲计算效率为代价换取内存节省的一种方式。由于 TensorFlow 原生版本没有提供对重计算的官方支持，使用 custom_gradient 机制来实现。
此外，采用了基于 BF16/FP16 的混合精度训练方法。用户可以在模型层面配置精度，将更高精度应用于关键组件，而在其他部分使用较低精度。这种方法在生产工作负载中显示出显著的优势，包括平均提升 18% 的吞吐量、减少 16% 的训练时间以及降低 18% 的内存使用量，密集层的内存使用量最多可减少 28%。
#### 3.6.3 KV Cache Serving
![输入图片说明](/imgs/2025-12-17/ZGSG4PQvmT0NDNjX.png)
为了在对多个候选物料进行评分时提高推理效率，引入了一种 KV 缓存机制，该机制将用户行为token与针对候选物料的全局token之间的注意力计算分离开来。由于用户序列在不同候选物料之间保持不变，其内部表示可以一次性计算并重复使用。

## 4 实验与分析：
### 4.1 Experimental Setting
字节抖音广告CVR预测系统进行评估，离线数据为130天52亿个样本，其中123天用于训练，7天用于预测。
对比模型：短序列方法包括 TWIN和 DIN（Recent50）。长序列方法包括 SumPooling、DIN、HSTU和 Transformer。

### 4.2 Overall Performance
#### 4.2.1 Comparison of existing methods
![输入图片说明](/imgs/2025-12-17/ar2Cq6hLoS2m2jHJ.png)
#### 4.2.1 Ablation study
![输入图片说明](/imgs/2025-12-17/iGfaAIwezYSdGNEC.png)



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY5NDE2OTMxNywtMjA2Nzg3OTc5NywxMD
czOTY0MDU5LDgwMzA1OTQzNSwxNjI5MTgzNzMsNTc1Nzc1ODAz
LDEwNzk0MjEyNzEsLTE2MjY2MjE2NTUsOTMxMTgzMzY1LDEyOD
YyMzgzNzksLTkxOTc4MTAyOF19
-->