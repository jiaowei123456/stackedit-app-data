# Wukong: Towards a Scaling Law for Large-Scale Recommendation
[原文链接]：([Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/pdf/2403.02545))

## 0 摘要：
Scaling Law在模型效果的可持续提升中起着关键作用。遗憾的是，迄今为止推荐模型并未展现出类似于大型语言模型中所观察到的此类定律，这是由于其扩展机制的低效性所致。这一局限性给将这些模型应用于日益复杂的现实世界数据集带来了重大挑战。在本文中，我们提出了一种完全基于堆叠FM（因子分解机）的有效网络架构，以及一种协同扩展策略，统称为“WuKong”，以在推荐领域建立缩放定律。WuKong的独特设计使其能够通过更高更宽的层轻松捕捉各种任意阶的交互。我们在六个公共数据集上进行了广泛的评估，结果表明WuKong在质量方面始终优于最先进的模型。此外，我们还在一个内部的大规模数据集上评估了WuKong的可扩展性。结果表明，WuKong在质量上仍优于最先进的模型，同时在模型复杂度上遵循着跨越两个数量级的缩放规律，其性能可扩展至每例超过 100 亿次浮点运算，而此前的技术则无法达到这一水平。

## 背景
1. 随着现代数据集呈指数级增长，scaling law可扩展性变得愈发重要。
2. 以往DL的RS通过增加emb维度和宽度来获得类似scaling law规律，但是仅仅扩展模型的emb层并不能增强其捕获越来越多特征之间复杂交互的能力，ROI较低。
3. 希望为推荐模型寻找一种替代的扩展机制。具体而言，希望设计出一种统一的架构，其质量能够随着数据集大小、计算资源和参数预算的增加而持续提升。


## 3 模型结构：
### 3.1 Overview：
![输入图片说明](/imgs/2025-12-27/Av7AKks5hSxBSJKu.png)

如图2所示，WuKong随后采用了**Interaction Stack**，这是一种统一的神经网络层堆栈，用于捕获emb之间的**指数级高阶交叉特征**。交互堆栈的灵感来自二进制指数的概念。交互堆栈中的每一层由**Factorization Machine Block**（FMB）和**Linear Compression Block**（LCB）组成。


### 3.2 Embedding Layer：
常见的Embedding层处理方式：多-hot 输入 → 嵌入表 → （sum）聚合
不管是离散还是连续特征都会通过Embedding层转为$d$维的emb，emb为$X_0 \in \mathbb{R}^{n \times d}$，也是后续模型的输入。

### 3.3  Interaction Stack


$X_{i+1} = \mathrm{LN}\left( \mathrm{concat}\left( \mathrm{FMB}_i(X_i), \mathrm{LCB}_i(X_i) \right) + X_i \right)$

### 3.4 Factorization Machine Block (FMB)
FMB实现高阶特征交互，文章中提到第$i$层可以获得$2^i$阶交叉特征，以下是第$i$层的网络结构：

$\mathrm{FMB}(X_i) = \mathrm{reshape}\left( \mathrm{MLP}\left( \mathrm{LN}\left( \mathrm{flatten}\left( \mathrm{FM}(X_i) \right) \right) \right) \right)$

其中FM有两种：
1. $\mathrm{FM}(X) = X X^\top$ 计算复杂度为$o(n^2d)$
2. $\mathrm{FM}(X) = X X^\top Y$  ，其中$Y$为维度为$n×k$可学习的矩阵，$k<<n$，计算复杂度为$o(ndk)$

PS：
1. FM和原始的FM还是不太一样的，没有包含可学习的交叉特征权重，而是直接通过MLP进行映射输出；
2. 这一部分的参数应该主要在MLP上面，交叉特征会随着层数的增加成指数级增加，如果MLP层不改变输入维度，那MLP的参数数量也会随层数成指数增加。

### 3.5 Linear Compress Block (LCB)
LCB简单地线性重组嵌入，而不增加交互顺序，这对于确保在整个层中保持交互顺序的不变性至关重要。具体来说，它保证第i交互层捕获范围从1到2i的交互顺序。LCB的操作描述如下：
$\mathrm{LCB}(X_i) = W_L X_i$

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
eyJoaXN0b3J5IjpbODMwNzY0MzA0LC0xNjk2NzQ2NDcsLTkyOT
gxMTMxNCw1ODE3ODc3ODcsNDUyNTQzNDk0LDIxMzYxNDA1MTcs
LTQ3NzI2MjIzNV19
-->