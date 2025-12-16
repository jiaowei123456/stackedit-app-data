# LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders
[原文链接]([LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/pdf/2505.04421))

## 0 摘要：
对超长用户行为序列进行建模对于确定工业推荐修复系统的长期和短期偏好至关重要。现有的解决方案通常依赖于两阶段检索或间接建模范式，导致上下游不一致和计算效率低下。在本文中，我们提出了一个用于gpu高效推荐器的长序列优化transformer结构**LONGER**。LONGER包含了(i)一个全局token机制，用于在长上下文内稳定注意力；（ii）一个带有轻量级inner transformer和混合注意策略的token合并模块，以降低二次复杂度；（iii）进行了一系列工程优化，包括混合精度训练和激活重计算、KV 缓存服务，以及用于统一基于 GPU 的密集和稀疏参数更新的全同步模型训练和服务框架。在字节跳动的广告和电商服务中，LONGER 在离线指标和在线 A/B 测试中始终优于强大的基线模型，验证了其持续的有效性和工业级scaling laws。目前，LONGER 已在字节跳动的数十个具有影响力的现实场景中得到验证和全面部署，为数十亿用户提供服务。

## 背景
1、简单的堆叠特征交互层，结构未修改，效果微弱甚至为负面。
2、DHEN和Wukong设计创新的深度神经网络结构以提高扩展性能。

## 1 论文解决的问题：
1、必须严格遵守严格的延迟限制，并支持极高的每秒查询数（QPS）。
2、原始rank模型的注意力机制主要在CPU计算时期提出，核心操作大多受内存限制而非计算限制，在现代 GPU 上，这导致了较差的 GPU 并行性以及极低的 MFU（模型运算次数利用率）。
3、架构应与硬件相匹配，以在现代 GPU 上实现最大化的MFU和计算吞吐量。
4、模型设计必须利用推荐数据的特性，例如异构特征空间以及数百个字段之间的个性化跨特征交互。

## 2 论文创新点：
1. Multi-head token mixing ：只通过无参数操作符获得跨token特征交互。该策略在性能和计算效率方面优于自注意机制。
2. Per-token feed-forward networks (FFNs)：通过为不同的特征子空间建模分配独立的参数，极大地扩展了模型容量，解决了特征空间间的控制问题。
3. Sparse Mixture-of-Experts (MoE)：通过针对不同的数据动态激活每个标记的特定子集专家，我们能够以最小的计算成本显著提高模型的容量。


## 3 模型结构：
### 3.1 整体框架：
![输入图片说明](/imgs/2025-12-13/iXt3rIjZqdbgMP4S.png)

输入为T个token，经过连续L个Rankmixer以及平均池化后输出，每个 RankMixer 块有两个主要组成部分：（1）多头token mixing；（2）每个token的per-token FFN（PFFN）层，如图所示。

$S_{n-1} = \operatorname{LN}\!\left( \operatorname{TokenMixing}(X_{n-1}) + X_{n-1} \right)$
$X_n = \mathrm{LN} \left( \mathrm{PFFN} \left( S_{n-1} \right) + S_{n-1} \right)$
### 3.2 输入层和特征token化：
1. 用户特征：包括用户 ID 及其他用户信息等
2. 物品特征：视频 ID、作者 ID 等
3. 序列特征：通过序列模块处理后的序列特征用于捕捉时间相关性
4. 交叉特征：用户侧与物品侧的交叉特征

Tokenization：为了实现高效的并行计算，不同维度的embedding必须转换为维度对齐的向量，这些向量被称为特征Token，这个过程称为Tokenization。

**最简单的策略是为每个特征分配一个embbeding，当特征为几百个时，每个token所分配的参数和计算量会衰减到很少，从而导致对重要特征的建模不足以及GPU核心的不充分利用。相反，token数量过少（例如仅一个token）会使模型结构退化为简单的深度神经网络（DNN），无法清晰地表示不同的特征空间，这可能会导致主导特征掩盖其他特征。**

为了解决这些问题，本文提出了一种基于语义的分词方法，结合领域知识将特征分组为几个语义连贯的簇。这些分组的特征依次连接成一个嵌入向量 $e_{\mathrm{input}} = \left[ e_1; e_2; \ldots; e_N \right]$，随后将其划分为具有固定维度大小的适当数量的标记。每个特征标记 $x_i ∈ R^D$ 捕获一组表示相似语义方面的特征嵌入。
$x_i = \mathrm{Proj}\left(e_{\mathrm{input}}\left[d \cdot (i - 1) : d \cdot i\right]\right), \quad i = 1, \ldots, T$

输出为 $x_i ∈ R^{T×D}$ ，T个token，每个token D维。

### 3.3 RankMixer Block
#### 3.3.1 Multi-head Token Mixing
先把每个token分成H个头：
$\left[ \mathbf{x}_t^{(1)} \parallel \mathbf{x}_t^{(2)} \parallel \cdots \parallel \mathbf{x}_t^{(H)} \right] = \mathrm{SplitHead}(\mathbf{x}_t)$

然后把T个token的每个h位置的头拼接起来（所有的多头操作都号称为了从multi-perspective解决任务，有没有文章能证明？）：
$\mathbf{s}^h = \left[ \mathbf{x}_1^h; \mathbf{x}_2^h; \ldots; \mathbf{x}_T^h \right]$

最后把拼接之后的H个$s^h$堆叠在一起，输出为$\mathbf{S} \in \mathbb{R}^{H \times \frac{TD}{H}}$。

原文中设置H=T，加上残差连接和归一化层后为：
$s_1, s_2, \ldots, s_T = \mathrm{LN}\!\left( \mathrm{TokenMixing}(x_1, x_2, \ldots, x_T) + (x_1, x_2, \ldots, x_T) \right)$

尽管自注意力机制在大型语言模型中表现出了极高的有效性，但我们发现它对于推荐系统而言效果并不理想。**在自注意力机制中，注意力权重是通过token的内积来计算的。这种方法在自然语言处理中效果良好，因为所有的token共享一个统一的embbeding空间。然而，在推荐任务中，特征空间本质上是异构的。在两个异构的语义空间之间计算内积相似度是极其困难的**——特别是在推荐系统中，用户和项目侧特征的 ID 空间可能包含数亿个元素。（逻辑是对的，所以行为序列建模一直用的是同语义空间的内积建模）（不过这个TokenMixing怎么这么像切牌啊，把token来回切）
#### 3.3.2 Per-token FFN
之前的 DLRM 和 DHEN 模型往往会在一个单一的交互模块中将来自多个不同语义空间的特征混合在一起，这可能会导致高频字段占据主导地位，从而掩盖低频或长尾信号，最终损害整体推荐质量。我们引入了一种参数独立的前馈网络架构，称为Per-token FFN。在传统的设计中，FFN 的参数在所有token中是共享的，但我们的方法对每个token都进行专门的变换，从而使得每个token有自己的FFN。

对于第 t 个令牌 $s_t$ ，每个令牌的 FFN 可以表示为
$\mathbf{v}_t = f_{\mathrm{pffn}}^{t,2} \left( \mathrm{Gelu} \left( f_{\mathrm{pffn}}^{t,1} (s_t) \right) \right)$
其中：
$f_{\mathrm{pffn}}^{t,i}(x) = x \mathbf{W}_{\mathrm{pffn}}^{t,i} + \mathbf{b}_{\mathrm{pffn}}^{t,i}$

输入为$\mathbf{s_t} \in \mathbb{R}^{\frac{TD}{H}}$，网络结构和传统的FFN网络差不多，先通过$f_{\mathrm{pffn}}^{t,1}$升维，然后通过$f_{\mathrm{pffn}}^{t,2}$降维，$Gelu(·)$是激活函数。

将per-token FFN模块总结为
$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_T = \mathrm{PFFN}(\mathbf{s}_1, \mathbf{s}_2, \ldots, \mathbf{s}_T)$

与参数全共享FFN相比，per-token FFN在保持计算复杂度不变的情况下，通过引入更多的参数来增强建模能力。（为什么，$\mathbf{W}_{\mathrm{pffn}}$如果是全参数共享的FFN参数应该是$T×D*T×kD$，per-token FFN参数应该是$D*kD*T$，感觉参数量变少了，有没有大佬帮忙推导一下？）

### 3.4 Sparse MoE in RankMixer
为了进一步提高ROI，我们可以将每个token的FFN替换为Sparse Mixture-of-Experts (MoE)，这样模型的容量就能增加，而计算成本则大致保持不变。然而，普通的稀疏专家混合模型（Sparse-MoE）在 RankMixer 中会表现不佳，原因在于：
（i）uniform k-expert routing：对前 k 个特征词的处理方式是同等对待所有特征token，这会浪费低信息特征token的资源并剥夺高信息特征token的资源，从而阻碍模型捕捉token之间的差异。
（ii）expert under-training：Per-token FFNs已经将参数乘以token的数量；再加上非共享专家会进一步增加专家的数量，导致路由高度不均衡且专家训练效果不佳。

文中提出两个解决方案：

ReLU Routing：为了使令牌拥有灵活的专家数量并保持可微性，我们用一个 ReLU 门控机制加上自适应 L1 惩罚来取代常见的 Topk + 指数化操作。

$G_{i,j} = \mathrm{ReLU}(h(s_i)), \quad \mathbf{v}_i = \sum_{j=1}^{N_e} G_{i,j} \, e_{i,j}(s_i)$
其中$N_e$是每个令牌的专家数量，$N_t$是令牌的数量。ReLU路由将激活更多的高信息令牌专家，提高参数效率。稀疏性由$L_{reg}$控制，其系数λ使平均活跃专家比率接近预算：
$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \lambda \, \mathcal{L}_{\mathrm{reg}}, \quad \mathcal{L}_{\mathrm{reg}} = \sum_{i=1}^{N_t} \sum_{j=1}^{N_e} G_{i,j}$

Dense-training / Sparse-inference (DTSI-MoE)：采用了$h_{train}$和$h_{infer}$两个路由器，$L_{reg}$仅用于$h_{infer}$。$h_{train}$和$h_{infer}$都在训练过程中更新，而在推理过程中只使用$h_{infer}$。事实证明，DS-MoE在降低推理成本的同时，使专家不会受到训练不足的困扰。

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
eyJoaXN0b3J5IjpbLTkxOTc4MTAyOF19
-->