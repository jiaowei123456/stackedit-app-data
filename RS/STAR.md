# One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
[原文链接](https://doi.org/10.1145/3383313.3412236)
## 0 摘要：
传统的行业推荐系统通常会使用单一领域的数据来训练模型，并服务于该单一领域。然而，一个大规模的商业平台往往包含多个领域，其推荐系统通常需要对多个领域进行点击率（CTR）预测。通常，不同的领域可能会有一些共同的用户群体和项目，而每个领域又可能有其独特的用户群体和项目。此外，即使是同一个用户在不同的领域中也可能有不同的行为。为了充分利用来自不同领域的所有数据，可以训练一个单一的模型来服务于所有领域。然而，单个模型难以捕捉各个领域的特征并很好地服务于所有领域。另一方面，为每个领域单独训练一个模型并不能充分利用来自所有领域的数据。在本文中，我们提出了星型拓扑自适应推荐器（STAR）模型，通过同时利用来自所有领域的数据来训练一个单一模型，捕捉每个领域的特征，并对不同领域的共性进行建模。从本质上讲，每个领域的网络由两个分解网络组成：一个由所有领域共享的中心网络，以及针对每个领域定制的领域特定网络。对于每个领域，我们将这两个分解网络结合起来，并通过将共享网络的权重与领域特定网络的权重进行元素相乘的方式生成一个统一的网络，尽管这两个分解网络也可以使用其他函数进行组合，这方面的研究还有待进一步开展。最重要的是，STAR 能够从所有数据中学习共享网络，并根据每个领域的特点调整领域特定参数。来自实际数据的实验结果验证了所提出的 STAR 模型的优越性。自 2020 年末以来，STAR 已被部署在阿里巴巴的展示广告系统中，点击率（CTR）提高了 8.0%，每千次展示收入（RPM）增加了 6.0%。
## 1 论文解决的问题：
* 负迁移（negative transfer）：MTL提出来的目的是为了不同任务，尤其是数据量较少的任务可以借助transfer learning（通过共享embedding，当然你也可以不仅共享embedding，再往上共享基层全连接网络等等这些很常见的操作）。但经常事与愿违，当两个任务之间的相关性很弱（比如一个任务是判断一张图片是否是狗，另一个任务是判断是否是飞机）或者非常复杂时，往往发生负迁移，即共享了之后效果反而很差，还不如不共享。
* 跷跷板现象：还是当两个task之间相关性很弱或者很复杂时，往往出现的现象是：一个task性能的提升是通过损害另一个task的性能做到的。这种现象存在很久，PLE论文里给它起了个非常贴切的名字『跷跷板』。
* MMOE通过门控网络根据输入来组合底层专家以处理任务差异，但忽略了专家之间的差异和相互作用，这在我们的工业实践中已被证明会引发跷跷板现象。
## 2 论文创新点：
* PLE 明确地将共享专家和任务特定专家分开，以减轻通用知识和任务特定知识之间有害的参数干扰。
* PLE 引入了多层专家和门控网络，并应用渐进式分离路由从较低层专家中提取更深层次的知识，并逐步在较高层分离任务特定参数。
## 3 相关工作：
* 直接的多塔线性组合，学习线性组合的权重；融合来自不同任务的表示。a，f
* MOE 通过门控网络将专家进行组合；MMOE 针对每个任务使用不同的门控机制；MRAN采用多头自注意力机制来在不同的特征集上学习不同的表示子空间。h
* 
![输入图片说明](/imgs/2025-07-08/4sKGptX6jkr7uNdd.png)
## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-08/gDJqongZ5GfUFQnO.png)
### 整体代码
```Python
task_fea = [emb for i in range(self.task_num + 1)] # task1 input ,task2 input,..taskn input, share_expert input  
for i in range(self.layers_num):  
    share_output=[expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]  
    task_output_list=[]  
    for j in range(self.task_num):  
        task_output=[expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]  
        task_output_list.extend(task_output)  
        mix_ouput=torch.cat(task_output+share_output,dim=1)  
        gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)  
        task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)  
    if i != self.layers_num-1:#最后一层不需要计算share expert 的输出  
        gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)  
        mix_ouput = torch.cat(task_output_list + share_output, dim=1)  
        task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)  
  
results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
```
### 4.1 专家网络
可以理解为每一层都有很多专家，专家可以分为专用专家(num_task)和通用专家(1)，每一个专家有自己的子全连接-专家个数。
#### 共享专家代码实现
```Python
share_output=[expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]] # 输入为（batch_size, input_dim），share_experts为layers_num层，每一层有shared_expert_num个全连接层——MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)，最后输出为（batch_size, 1, bottom_mlp_dims[i]），有shared_expert_num个
```
#### 特殊专家代码实现
```Python
task_output=[expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]] # 输入为（batch_size, input_dim），task_experts为layers_num层，每一层有specific_expert_num个全连接层——MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)，最后输出为（batch_size, 1, bottom_mlp_dims[i]），有specific_expert_num个。注：特殊专家网络mlp数量为layers_num*task_num*specific_expert_num
```
#### 特殊专家门控代码实现
```Python
gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1) # 每一个任务都有一个对应的门控结果，因此门控网络数量为layers_num*task_num，每一个网络为：torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1))，因此输出为（batch_size, 1, shared_expert_num + specific_expert_num）
```
#### 特殊专家加权输出
```Python
mix_ouput = torch.cat(task_output + share_output,dim=1)   #shared_expert_num个共享专家，specific_expert_num个特殊专家拼接
task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1) # 加权输出，输出维度为（batch_size, 1, bottom_mlp_dims[i]）
```
### 4.2 中间层的混合专家加权输出（非最后一层）
```Python
if i != self.layers_num-1:#最后一层不需要计算share expert 的输出  
    gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)  #（batch_size, 1, shared_expert_num + specific_expert_num*task_num）
    mix_ouput = torch.cat(task_output_list + share_output, dim=1)   #（batch_size, shared_expert_num + specific_expert_num*task_num，bottom_mlp_dims[i]）
    task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)  #（batch_size,1，bottom_mlp_dims[i]）
```
### 4.3 最终每个任务加一个全连接层预测最终输出
```Python
results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)] #使用sigmoid作为激活函数。输出（batchsize，num_task）
```

## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ3Mzk4NDkyNV19
-->