# Deep & Cross Network for Ad Click Predictions (DCNv1)
[原文链接](https://arxiv.org/abs/1708.05123)
## 0 摘要：
点击率预测是推荐系统（RS）中的核心任务之一。它会为每个用户-项目对预测出个性化的点击概率。最近，研究人员发现，通过考虑用户行为序列（尤其是长期用户行为序列），CTR 模型的性能可以得到极大的提升。一份关于某电子商务网站的报告表明，在过去的 5 个月里，有 23%的用户进行了超过 1000 次的点击。尽管有许多研究工作专注于对用户行为序列进行建模，但由于现实世界系统中严格的推理时间限制，很少有工作能够处理长期用户行为序列。为此，提出了两阶段方法以突破性能限制。在第一阶段，设计了一个辅助任务，从长期用户行为序列中检索出前 k 个相似项目。在第二阶段，将经典的注意力机制应用于候选项目和第一阶段中选择的 k 个项目之间。然而，检索阶段和主要的点击率任务之间存在信息差距。这种目标差异会极大地削弱长期用户序列带来的性能提升。在本文中，受 Reformer 的启发，我们提出了一种名为 ETA（端到端目标注意力）的局部敏感哈希方法，该方法能够大幅降低训练和推理成本，并使具有长期用户行为序列的端到端训练成为可能。无论是离线实验还是在线实验都证实了我们模型的有效性。我们将 ETA 应用于一个大规模的真实世界电子商务系统中，并与一个两阶段的长用户序列点击率模型相比，在商品总价值（GMV）方面实现了额外 3.1%的提升。
## 1 论文解决的问题：
* 面向网络规模推荐系统的数据大多为离散和分类数据，这导致了特征空间庞大且稀疏，给特征探索带来了巨大挑战。这使得大多数大规模系统只能采用线性模型，如逻辑回归。
* 相比之下，交叉特征已被证明能够显著提升模型的表达能力。但不幸的是，要识别这些特征往往需要手动特征工程或进行详尽的搜索；此外，对于未知的特征交互关系进行推广也是困难的。
## 2 论文创新点：
* 引入了一种新颖的神经网络结构——交叉网络，它能够自动有效地应用特征交叉操作。该交叉网络由**多个层**组成，其中最高程度的相互作用显然由层的深度所决定。每一层都基于**现有的相互作用**生成**更高阶的相互作用**，并**保留前一层的相互作用**。
## 3 相关工作：
* 因子分解机（FMs）
## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-06/50vPq9ZicRMCxKcP.png)
### 4.1 Embeddding
我们考虑具有稀疏和稠密特征的输入数据。稀疏特征通常被编码为one-hot向量，然后进行嵌入编码。稠密特征直接归一化保留。
### 4.2 Cross Network
该交叉网络的核心思想是有效地应用显式特征交叉。交叉网络由交叉层组成，每层有如下公式：
$$x_{l+1}=x_{0}x_{l}^Tw_{l}+b_{l}+x_{l}$$
#### 代码实现
```Python
class CrossNet(nn.Module):  
    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):  
        super(CrossNet, self).__init__()  
        self.layer_num = layer_num  
        self.parameterization = parameterization  
        self.kernels = nn.Parameter(torch.Tensor(layer_num, in_features, 1)) #  创建一个可学习的参数张量，形状为 (layer_num, in_features, 1) 
        self.bias = nn.Parameter(torch.Tensor(layer_num, in_features, 1))  
  
        for i in range(self.kernels.shape[0]):  
            nn.init.xavier_normal_(self.kernels[i]) # Xavier 初始化（也称为 Glorot 初始化）旨在解决深度神经网络中梯度消失和梯度爆炸的问题。  
        for i in range(self.bias.shape[0]):  
            nn.init.zeros_(self.bias[i])  
  
        self.to(device)  
  
    def forward(self, inputs):  
        x_0 = inputs.unsqueeze(2)  # [B, D, 1]  
        x_l = x_0  
        for i in range(self.layer_num):  
        # 表示在 x_l 的第 1 维度（即 in_features）和 self.kernels[i] 的第 0 维度（即 in_features）之间进行求和运算（类似矩阵乘法中的内积操作）。
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=[[1], [0]])  # [B, 1, 1] 
            dot_ = torch.matmul(x_0, xl_w)   # 进行 矩阵乘法（matmul）运算
            x_l = dot_ + self.bias[i] + x_l  # 类似残差连接
        x_l = torch.squeeze(x_l, dim=2)  
        return x_l
```
#### 复杂度分析
设Lc表示交叉层数，d表示输入维数。则交叉网络中涉及的参数个数为：
$$d×Lc×2$$交叉网络的时间和空间复杂度与输入维度呈线性关系。因此，与深度网络相比，交叉网络引入的复杂度几乎可以忽略不计，使得深度卷积网络的整体复杂度与传统深度神经网络处于同一水平。这种高效性得益于 x0xT  l 的秩为一的特性，这使我们能够无需计算或存储整个矩阵即可生成所有交叉项。
### 4.3 Deep Network
线性层+bn层+激活函数+dropout层
```Python
class DNN(nn.Module):  
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,  
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):  
        super(DNN, self).__init__()  
        self.dropout_rate = dropout_rate  
        self.dropout = nn.Dropout(dropout_rate)  
        self.seed = seed  
        self.l2_reg = l2_reg  
        self.use_bn = use_bn  
        if len(hidden_units) == 0:  
            raise ValueError("hidden_units is empty!!")  
        if inputs_dim > 0:  
            hidden_units = [inputs_dim] + list(hidden_units)  
        else:  
            hidden_units = list(hidden_units)  
  
        self.linears = nn.ModuleList(  
            [nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units) - 1)])  
  
        if self.use_bn:  
            self.bn = nn.ModuleList(  
                [nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units) - 1)])  
  
        self.activation_layers = nn.ModuleList(  
            [activation_layer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units) - 1)])  
  
        for name, tensor in self.linears.named_parameters():  
            if 'weight' in name:  
                nn.init.normal_(tensor, mean=0, std=init_std)  
  
        self.to(device)  
  
    def forward(self, inputs):  
        deep_input = inputs  
        for i in range(len(self.linears)):  
            # print(f"i:{i}, deep_input size:{deep_input.size()}")  
            fc = self.linears[i](deep_input)  
  
            if self.use_bn and fc.size()[0] > 1:  
                fc = self.bn[i](fc)  
  
            fc = self.activation_layers[i](fc)  
  
            fc = self.dropout(fc)  
            deep_input = fc  
        return deep_input
```
#### 复杂度分析
为了简化起见，我们假设所有深层结构的大小都相同。设 Ld 表示深层结构的数量，m 表示每个深层结构的大小。那么，深层网络中的参数数量为
$$d × m + m + (m_2 + m) × (L_d − 1).$$第一层参数是d × m + m，后面Ld-1层是(m_2 + m) × (L_d − 1)

## 5 实验与分析：
-   多项式拟合：我们证明，在只有O(d)个参数的情况下，交叉网络包含了所有出现在同一次多项式中的交叉项，并且每个项的系数彼此不同。
-   FM的泛化：因此，交叉网络将参数共享的概念从单层扩展到了多层以及高阶交叉项。需要注意的是，与高阶 FM 不同，交叉网络中的参数数量仅随输入维度线性增长。
-   高效映射：每个交叉层以一种有效的方式将x0和xl之间的所有成对相互作用投影回输入维度。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0ODk1OTg0ODFdfQ==
-->