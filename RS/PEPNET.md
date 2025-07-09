# One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
[原文链接](https://doi.org/10.1145/3383313.3412236)
## 0 摘要：
传统的行业推荐系统通常会使用单一领域的数据来训练模型，并服务于该单一领域。然而，一个大规模的商业平台往往包含多个领域，其推荐系统通常**需要对多个领域进行点击率（CTR）预测**。通常，不同的领域可能会有一些共同的用户群体和项目，而每个领域又可能有其独特的用户群体和项目。此外，即使是同一个用户在不同的领域中也可能有不同的行为。为了充分利用来自不同领域的所有数据，可以训练一个单一的模型来服务于所有领域。然而，单个模型难以捕捉各个领域的特征并很好地服务于所有领域。另一方面，为每个领域单独训练一个模型并不能充分利用来自所有领域的数据。在本文中，我们提出了星型拓扑自适应推荐器（Star Topology Adaptive Recommender, STAR）模型，通过同时利用来自所有领域的数据来训练一个单一模型，捕捉每个领域的特征，并对不同领域的共性进行建模。从本质上讲，**每个领域的网络由两个分解网络组成：一个由所有领域共享的中心网络，以及针对每个领域定制的领域特定网络**。对于每个领域，我们将这两个分解网络结合起来，并通过将共享网络的权重与领域特定网络的权重进行元素相乘的方式生成一个统一的网络，尽管这两个分解网络也可以使用其他函数进行组合，这方面的研究还有待进一步开展。最重要的是，STAR 能够从所有数据中学习共享网络，并根据每个领域的特点调整领域特定参数。来自实际数据的实验结果验证了所提出的 STAR 模型的优越性。自 2020 年末以来，STAR 已被部署在阿里巴巴的展示广告系统中，点击率（CTR）提高了 8.0%，每千次展示收入（RPM）增加了 6.0%。
## 1 论文解决的问题：

## 2 论文创新点：

## 3 相关工作：

## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-09/UtZLPMiTfAmVX2li.png)
### 4.1 门控权重生成器实现
```Python
class GateNU:  
    def __init__(self,  
                 hidden_units,  
                 gamma=2.,  
                 l2_reg=0.):  
        assert len(hidden_units) == 2  
        self.hidden_units = hidden_units  
        self.gamma = gamma  
        self.l2_reg = l2_reg # 防止过拟合，提升泛化能力  
  
    def __call__(self, inputs):  
        output = tf.layers.dense(inputs, self.hidden_units[0], activation="relu",  
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)) # 引入非线性能力，增强表达力  
        output = tf.layers.dense(output, self.hidden_units[1], activation="sigmoid", # 输出范围 [0, 1]，可解释为重要性权重或注意力分数  
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))  
        return self.gamma * output # 放大输出值，增强 gate 的影响力度（类似 attention 中的温度系数）
```
### 4.2 共享专家代码实现
```Python
class EPNet:  
    def __init__(self,  
                 hidden_units,  
                 l2_reg=0.):  
  
        self.gate_nu = GateNU(hidden_units=hidden_units, l2_reg=l2_reg)  
  
  
    def __call__(self, domain, emb):  
        # domain: 当前任务/场景的领域特征（persona），形状为 [B, D]        # emb: 输入嵌入向量，通常是共享特征或上下文特征，形状为 [B, E]        # 使用 tf.stop_gradient 冻结 emb 的梯度，防止 gate 影响其更新  
        # 输出 gate 权重张量，形状为 [B, E]（和 emb 同维度） 这里形状由hidden_units确定  
        # 使用 gate 权重对原始嵌入进行 element-wise 相乘  
        return self.gate_nu(tf.concat([domain, tf.stop_gradient(emb)], axis=-1)) * emb
```
## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbNjY0ODIyNTI5LDEzMDg0NjE0OTMsMzUwNT
U3MjUxLDc3NzgxMjIyM119
-->