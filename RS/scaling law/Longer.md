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
其中G表示全局token长度为m，H表示序列token长度为L
然后获取查询矩阵，其中G为全部的全局token，Hs为抽样获得的k个序列token，注意这个k和3.4的K不同，抽样的策略效果后面有做实验进行比较。
$\mathbf{O} = [\mathbf{G}; \mathbf{H}_s]$

#### 3.5.2 Cross-Causal Attention (First Layer)
![输入图片说明](/imgs/2025-12-17/KHWNHKaaay6lGRd2.png)
第一层是交叉因果注意力计算：
$Q = O W_Q, \quad K = R W_K, \quad V = R W_V$

$\mathbf{Q} = \mathbf{O} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{R} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{R} \mathbf{W}_V$
其中O是目标query，R是key，value。M是一个右上值全为0左下值全为-inf的三角矩阵。
$\mathbf{M}_{i,j} = \begin{cases} 0, & \text{if } j \geq i, \text{ where } \{i, j\} \in [1, m + L] \\ -\infty, & \text{otherwise} \end{cases}$
因果掩码M设计一方面保持了序列中各元素之间的时间相关性。另一方面，它确保了序列对候选元素的不可见性，从而实现了 KV 缓存服务机制。在计算完注意力值后，结果会通过前馈网络（FFN）进行进一步处理。

#### 3.5.3 Self-Causal Attention (Subsequent Layers).
![输入图片说明](/imgs/2025-12-17/kRO2ZaF8Pi7GY9QQ.png)
在交叉因果注意力层之后，后续的层由N个自因果注意力层组成。这些层专注于学习采样token序列内的内部关系。每个自因果注意力层之后都接有一个FFN。自因果注意力机制的计算使用了类似的公式：
$\mathbf{Q} = \mathbf{O} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{R} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{R} \mathbf{W}_V$
其中Q, K, V全为上一层的输出经过一个线性层。

### 3.6 Training and Deployment Optimization
#### 3.6.1 Training Framework
训练流程始于以批处理或流式形式进行的数据摄取，随后通过 Fountain 模块进行预处理。处理后的训练数据随后被分发到多个 GPU 运行器中，在那里密集和稀疏参数会同步更新。该框架的一个关键特性是其统一的参数存储和训练架构。无论是密集参数还是稀疏参数，都可在 GPU 机器上同步进行存储和更新，从而无需外部参数服务器组件。为了更好地适应推荐系统中的特征分布模式，该框架采用了分层存储系统来处理稀疏嵌入，能够高效支持大型嵌入表。在这一设计中，高频特征存储在高带宽的 GPU 内存（HBM）中，中频特征存放在 CPU 主内存（MEM）中，低频特征则被卸载到本地固态硬盘（SSD）上。这种分层存储布局经过优化，以匹配推荐数据的访问特性，实现了延迟、吞吐量和容量之间的实际平衡。核心创新在于将计算和参数存储完全集中于 GPU 机器上，从而减少通信开销和内存传输延迟。这带来了更高的训练吞吐量、更低的滞留率和更强的收敛稳定性。
#### 3.6.2 Mixed Precision Training and Recompute



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
eyJoaXN0b3J5IjpbLTEwMTY3MTExMzksMTA3Mzk2NDA1OSw4MD
MwNTk0MzUsMTYyOTE4MzczLDU3NTc3NTgwMywxMDc5NDIxMjcx
LC0xNjI2NjIxNjU1LDkzMTE4MzM2NSwxMjg2MjM4Mzc5LC05MT
k3ODEwMjhdfQ==
-->