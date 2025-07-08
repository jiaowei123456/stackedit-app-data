# Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
[原文链接](https://dl.acm.org/doi/10.1145/3219819.3220007)
## 0 摘要：
基于神经网络的多任务学习已在诸如推荐系统等许多现实世界的大型应用中成功运用。例如，在电影推荐方面，除了为用户提供他们可能购买和观看的电影外，系统还可能优化用户在观看后的喜好程度。通过多任务学习，我们旨在构建一个能够同时学习这些多个目标和任务的单一模型。然而，常用的多任务模型的预测质量往往对**任务之间的关系**很敏感。因此，研究任务特定目标与任务间关系之间的建模权衡至关重要。在本研究中，我们提出了一种新颖的多任务学习方法——多门控专家混合模型（MMoE），该模型能够从数据中显式地**学习任务关系**。我们通过在所有任务中共享专家子模型，并同时训练一个门控网络来优化每个任务，将专家混合模型（MoE）结构应用于多任务学习。为了在具有不同任务相关程度的数据上验证我们的方法，我们首先将其应用于一个合成数据集，在该数据集中我们可以控制任务相关程度。我们表明，当任务**相关性较低时，所提出的方法比基准方法表现更优**。我们还表明，MMoE 结构会带来额外的可训练性优势，这取决于训练数据和模型初始化中的不同随机程度。此外，我们通过包括二元分类基准和谷歌的大规模内容推荐系统在内的实际任务，展示了 MMoE 带来的性能提升。
## 1 论文解决的问题：
* 许多基于深度神经网络的多任务学习模型对诸如数据分布差异以及任务之间的关系等因素非常敏感，由于任务之间的差异所导致的内在冲突实际上可能会损害至少某些任务的预测结果，尤其是在模型参数在所有任务中被广泛共享的情况下。
## 2 论文创新点：
* MMoE 明确地对任务关系进行建模，并学习任务特定的功能以利用共享表示。它允许参数自动分配，以捕捉共享的任务信息或任务特定的信息，从而避免了为每个任务添加大量新参数的需求。
* 我们还发现 MMoE 更易于训练，并在多次运行中收敛到更好的损失值。这与最近的发现有关，即调制和门控机制可以提高训练非凸深度神经网络的可训练性
## 3 相关工作：
* Caruana提出的多任务学习网络
* 
## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-08/2mfzuxK6OdtwCFwc.png)
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
eyJoaXN0b3J5IjpbNjg0ODY1MSwtMTI1NzQwOTQ2OCwtMTIzMD
E2NTI4NCw3OTU1NzI1NCwxMjM3MTE3NzAsLTg1MTk5OTcxNCwt
MTc4MzY5MzkyMiw2NjE2NzkyMl19
-->