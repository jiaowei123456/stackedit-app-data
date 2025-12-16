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
1. 一种用于gpu高效推荐的长序列优化transformer结构。它通过优化transformer结构，以端到端的方式将用户序列建模长度扩展到10,000(TWINv2?)。
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
原始transformer的二次注意力计算复杂度为$$




### 3.5 Scaling Up Directions
RankMixer 本质上是一种高度并行且可扩展的架构。其参数数量和计算成本可以通过四个相互垂直的维度进行扩展：令牌数量 T、模型宽度 D、层数 L 和专家数量 E。对于全密集激活版本，一个样本的参数数量和前向计算浮点运算次数可以计算为：
$\#\mathrm{Param} \approx 2kLT D^2, \quad \mathrm{FLOPs} \approx 4kLT D^2$

## 4 实验与分析：
### 4.1 实验设置
#### 4.4.1 Datasets and Environment
离线实验：基于抖音推荐系统的训练数据进行，300+特征，两周数据训练。
#### 4.4.2 Evaluation Metrics
AUC、UAUC、MFU
Finish/Skip AUC/UAUC：“finish=1/0”或“skip=1/0”标签表示用户是否在短时间内完成了视频观看或转到下一个视频。
#### 4.4.2 Baselines
DLRM-MLP、DCNv2、RDCN、AutoInt、Hiformer、DHEN、Wukong
### 4.2 Comparison with SOTA methods
不同模型比较，RankMixer效果最优：
![输入图片说明](/imgs/2025-12-15/Kz5s7WXsvZkGX1kI.png)

### 4.3 Scaling Laws of different models
![输入图片说明](/imgs/2025-12-15/qVNhNnokzfSIBQLG.png)

在本文的实验中，观察到了与 LLM 扩展规律相同的结论：模型质量主要与参数总数相关，不同的扩展方向（深度 L、宽度 D、令牌 T）几乎能产生相同的性能。**从计算效率的角度来看，更大的隐藏层维度会产生更大的矩阵乘法，从而比增加更多层实现更高的 MFU，最终的配置（100M 和 1B）分别为（D=768，T=16，L=2）和（D=1536，T=32，L=2）**
### 4.4 Ablation Study

![输入图片说明](/imgs/2025-12-15/kH4lRzLYLR7YQXI4.png)
All-Concat-MLP：将所有token进行连接，并通过一个大型MLP对其进行处理，然后再将其拆分成相同数量的token。
All-Share：不进行拆分，所有的输入向量共享并喂到每个per-token FFN类似于MoE。
Self-Attention:在token之间应用自注意力机制进行路由。
### 4.5 Sparse-MoE Scalability and Expert Balance
![输入图片说明](/imgs/2025-12-15/eWSDSUz96x2b92Zx.png)
Scalability. 上图绘制了SMoE 的离线 AUC 增益与稀疏度的关系。
1. 原始 SMoE 的性能随着激活的专家数量减少而单调下降，这说明了专家不平衡和训练不足的问题。
2. 添加balance loss可减少相对于原始 SMoE 的性能下降，但仍不如 DTSI(Dense-training / Sparse-inference) + ReLU 版本，因为问题主要在于专家训练而非路由器。

Expert balance and diversity 普通稀疏多专家模型常常会遭遇专家失衡的问题。图文证明，将 DTSI与 ReLU 路由相结合能够有效解决这一问题：Dense-training确保大多数专家都能获得足够的梯度更新。ReLU 路由使激活比例在各个token之间动态变化——图中显示的激活比例会根据其信息内容自适应地变化，这与推荐数据的多样化且高度动态的分布非常吻合。

### 4.6 Online Serving cost
做到了参数量提升70倍，但是推理耗时几乎不变，耗时计算公式：
$\text{Latency} = \frac{\#\text{Param} \times \text{FLOPs/Param ratio}}{\text{MFU} \times (\text{Theoretical Hardware FLOPs})}$

相比与原始online模型，延迟变化因素影响表：
![输入图片说明](/imgs/2025-12-15/H6jUiL03rt4smeof.png)

其实只需要看FLOPs、MFU以及Hardware FOLPs就可以，分别升高了20倍，10倍，2倍，所以前者被后两者抵消了。

MFU：如表 6 所示，MFU 表示机器计算的利用率。通过采用大型 **GEMM shape、良好的并行拓扑结构（将并行的每个令牌的 FFN 融合为一个内核）以及降低内存带宽成本和开销**，RankMixer 将 MFU 提高了近 10 倍，使模型从内存受限状态转变为计算受限状态。（感觉这个才是关键，其次没想到抖音居然之前用的是全精）

### 4.7 Online Performance
在抖音与抖音极速版上面进行了8个月的AB实验与反转实验。
推荐侧效果：很好，特别是在低活用户上面，**不过抖音极速版的comment的提升反而在高活用户更好，这是为什么？**
![输入图片说明](/imgs/2025-12-15/Ud8ZgNJAVaE1pLwR.png)
广告侧效果：也很好。
![输入图片说明](/imgs/2025-12-15/p8K56RwBUuUC71nm.png)

<!--stackedit_data:
eyJoaXN0b3J5IjpbOTI4MzQ3NjQ0LDU3NTc3NTgwMywxMDc5ND
IxMjcxLC0xNjI2NjIxNjU1LDkzMTE4MzM2NSwxMjg2MjM4Mzc5
LC05MTk3ODEwMjhdfQ==
-->