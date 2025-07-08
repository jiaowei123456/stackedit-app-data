# Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
[原文链接](https://dl.acm.org/doi/10.1145/3219819.3220007)
## 0 摘要：
多任务学习（MTL）已成功应用于众多推荐应用中。然而，由于现实世界推荐系统中任务间的复杂且相互竞争的关联性，MTL 模型往往会因**负迁移**而出现性能退化。此外，通过对最先进的 MTL 模型进行广泛的实验，我们观察到了一种有趣的**跷跷板**现象：一个任务的性能往往通过损害其他任务的性能而得到提升。为了解决这些问题，我们提出了一种具有新颖共享结构设计的渐进分层提取（PLE）模型。PLE 明确地将共享组件和任务特定组件分开，并采用渐进式路由机制逐步提取和分离更深层次的语义知识，从而在一般设置中提高联合表示学习的效率以及任务间的信息路由效率。我们将 PLE 应用于复杂相关性和正常相关性任务中，涵盖从两个任务案例到多任务案例等多个方面，所使用的数据集来自腾讯的现实世界视频推荐数据集，包含 10 亿个样本。结果表明，在不同的任务相关性和任务组规模下，PLE 都显著优于最先进的多任务学习模型。此外，在腾讯大规模内容推荐平台上的在线评估显示，与最先进的多任务学习模型相比，PLE 的浏览量提高了 2.23%，观看时间提高了 1.84%，这是一个显著的改进，证明了 PLE 的有效性。最后，在公共基准数据集上的大量离线实验表明，PLE 不仅适用于推荐场景，还可以应用于各种其他场景以消除跷跷板现象。PLE 现已成功部署到腾讯的在线视频推荐系统中。
## 1 论文解决的问题：
* 负迁移（negative transfer）：MTL提出来的目的是为了不同任务，尤其是数据量较少的任务可以借助transfer learning（通过共享embedding，当然你也可以不仅共享embedding，再往上共享基层全连接网络等等这些很常见的操作）。但经常事与愿违，当两个任务之间的相关性很弱（比如一个任务是判断一张图片是否是狗，另一个任务是判断是否是飞机）或者非常复杂时，往往发生负迁移，即共享了之后效果反而很差，还不如不共享。
* 跷跷板现象：还是当两个task之间相关性很弱或者很复杂时，往往出现的现象是：一个task性能的提升是通过损害另一个task的性能做到的。这种现象存在很久，PLE论文里给它起了个非常贴切的名字『跷跷板』。
* MMOE [13] 通过门控网络根据输入来组合底层专家以处理任务差异，但忽略了专家之间的差异和相互作用，这在我们的工业实践中已被证明会引发跷跷板现象。
## 2 论文创新点：

## 3 相关工作：

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
eyJoaXN0b3J5IjpbNzQzMTQwMTcxLDE3OTU3NTAyMzAsMjA4MD
U2MTYzNF19
-->