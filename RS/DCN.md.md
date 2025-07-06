# Deep & Cross Network for Ad Click Predictions (DCNv1)
[原文链接](https://arxiv.org/abs/1708.05123)
## 0 摘要：
特征工程一直是许多预测模型取得成功的关键因素。然而，这一过程并不简单，通常需要人工特征工程或进行详尽的搜索。深度神经网络能够自动学习特征之间的相互作用；然而，它们只是隐式地生成所有这些相互作用，并且**不一定在学习所有类型的交叉特征方面效率很高**。在本文中，我们提出了深度与交叉网络（DCN），它保留了深度神经网络模型的优点，并且还引入了一个更高效的新型交叉网络，能够更有效地学习某些有界度的特征相互作用。特别是，DCN 在**每一层都明确地应用特征交叉**，**无需人工特征工程**，并且对深度神经网络模型的**额外复杂性影响极小**。我们的实验结果表明，在点击率预测数据集和密集分类数据集上，与最先进的算法相比，DCN 在模型准确性和内存使用方面都具有优越性。
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
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=[[1], [0]])  # [B, 1, 1]  
            dot_ = torch.matmul(x_0, xl_w)  
            x_l = dot_ + self.bias[i] + x_l  
        x_l = torch.squeeze(x_l, dim=2)  
        return x_l
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5ODA0NzEzNTksLTc5ODMwNjU2NCwtMT
g5NjI4NjE1NSwtMTMxMDkyMDA2NSwzOTAxODE0NzgsMjYyNDkz
NTkzLDQ0MDkwNTYxOV19
-->