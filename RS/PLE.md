# Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
[原文链接](https://dl.acm.org/doi/10.1145/3219819.3220007)
## 0 摘要：
基于神经网络的多任务学习已在诸如推荐系统等许多现实世界的大型应用中成功运用。例如，在电影推荐方面，除了为用户提供他们可能购买和观看的电影外，系统还可能优化用户在观看后的喜好程度。通过多任务学习，我们旨在构建一个能够同时学习这些多个目标和任务的单一模型。然而，常用的多任务模型的预测质量往往对**任务之间的关系**很敏感。因此，研究任务特定目标与任务间关系之间的建模权衡至关重要。在本研究中，我们提出了一种新颖的多任务学习方法——多门控专家混合模型（MMoE），该模型能够从数据中显式地**学习任务关系**。我们通过在所有任务中共享专家子模型，并同时训练一个门控网络来优化每个任务，将专家混合模型（MoE）结构应用于多任务学习。为了在具有不同任务相关程度的数据上验证我们的方法，我们首先将其应用于一个合成数据集，在该数据集中我们可以控制任务相关程度。我们表明，当任务**相关性较低时，所提出的方法比基准方法表现更优**。我们还表明，MMoE 结构会带来额外的可训练性优势，这取决于训练数据和模型初始化中的不同随机程度。此外，我们通过包括二元分类基准和谷歌的大规模内容推荐系统在内的实际任务，展示了 MMoE 带来的性能提升。
## 1 论文解决的问题：
* 负迁移（negative transfer）：MTL提出来的目的是为了不同任务，尤其是数据量较少的任务可以借助transfer learning（通过共享embedding，当然你也可以不仅共享embedding，再往上共享基层全连接网络等等这些很常见的操作）。但经常事与愿违，当两个任务之间的相关性很弱（比如一个任务是判断一张图片是否是狗，另一个任务是判断是否是飞机）或者非常复杂时，往往发生负迁移，即共享了之后效果反而很差，还不如不共享。
* 跷跷板现象：还是当两个task之间相关性很弱或者很复杂时，往往出现的现象是：一个task性能的提升是通过损害另一个task的性能做到的。这种现象存在很久，PLE论文里给它起了个非常贴切的名字『跷跷板』。

## 2 论文创新点：
* MMoE 明确地对任务关系进行建模，并学习任务特定的功能以利用共享表示。它允许参数自动分配，以捕捉共享的任务信息或任务特定的信息，从而避免了为每个任务添加大量新参数的需求。
* 我们还发现 MMoE 更易于训练，并在多次运行中收敛到更好的损失值。这与最近的发现有关，即调制和门控机制可以提高训练非凸深度神经网络的可训练性
## 3 相关工作：
* Caruana提出的多任务学习网络
* 
## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-08/2mfzuxK6OdtwCFwc.png)
### 4.1 MoE网络
#### 代码实现
```Python
self.expert_kernels = self.add_weight(  
    name='expert_kernel',  
    shape=(input_dimension, self.units, self.num_experts),  
    initializer=self.expert_kernel_initializer,  
    regularizer=self.expert_kernel_regularizer,  
    constraint=self.expert_kernel_constraint,  
)

# f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper   
expert_outputs = tf.tensordot(a=inputs, b=self.expert_kernels, axes=1) # 输入(batch_size, input_dimension)的最后一维和权重(input_dimension,units, num_experts)的第一维点积(batch_size, units, num_experts)  
# Add the bias term to the expert weights if necessary  
if self.use_expert_bias:  
    expert_outputs = K.bias_add(x=expert_outputs, bias=self.expert_bias)  
expert_outputs = self.expert_activation(expert_outputs)
```
### 4.2 Gate网络
#### 代码实现
```Python
self.gate_kernels = [self.add_weight(  
    name='gate_kernel_task_{}'.format(i),  
    shape=(input_dimension, self.num_experts),  
    initializer=self.gate_kernel_initializer,  
    regularizer=self.gate_kernel_regularizer,  
    constraint=self.gate_kernel_constraint  
) for i in range(self.num_tasks)] #产生num_tasks个门控权重，每个权重大小为（input_dimension, self.num_experts）

# g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper    
for index, gate_kernel in enumerate(self.gate_kernels):  # 循环num_tasks次
    gate_output = K.dot(x=inputs, y=gate_kernel)  # 每个门控网络生成num_experts个注意力。
    # Add the bias term to the gate weights if necessary  
    if self.use_gate_bias:  
        gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])  
    gate_output = self.gate_activation(gate_output) # 激活函数为softmax 
    gate_outputs.append(gate_output) # gate_outputs列表长度为num_tasks，每一元素为（batch_size, num_experts）
```
### 4.3 Gate加权输出
#### 代码实现
```Python
# f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))   
for gate_output in gate_outputs:  
    expanded_gate_output = K.expand_dims(gate_output, axis=1)   #（batch_size, 1, num_experts）
    weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)  #  (batch_size, units, num_experts) * (batch_size, units, num_experts) 对应位置元素相乘
    final_outputs.append(K.sum(weighted_expert_output, axis=2))
```

## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA4MDU2MTYzNF19
-->