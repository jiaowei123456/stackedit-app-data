# Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
[原文链接](https://dl.acm.org/doi/10.1145/3219819.3220007)
## 0 摘要：
多任务学习（MTL）已成功应用于众多推荐应用中。然而，由于现实世界推荐系统中任务间的复杂且相互竞争的关联性，MTL 模型往往会因**负迁移**而出现性能退化。此外，通过对最先进的 MTL 模型进行广泛的实验，我们观察到了一种有趣的**跷跷板**现象：一个任务的性能往往通过损害其他任务的性能而得到提升。为了解决这些问题，我们提出了一种具有新颖共享结构设计的渐进分层提取（PLE）模型。PLE 明确地将共享组件和任务特定组件分开，并采用渐进式路由机制逐步提取和分离更深层次的语义知识，从而在一般设置中提高联合表示学习的效率以及任务间的信息路由效率。我们将 PLE 应用于复杂相关性和正常相关性任务中，涵盖从两个任务案例到多任务案例等多个方面，所使用的数据集来自腾讯的现实世界视频推荐数据集，包含 10 亿个样本。结果表明，在不同的任务相关性和任务组规模下，PLE 都显著优于最先进的多任务学习模型。此外，在腾讯大规模内容推荐平台上的在线评估显示，与最先进的多任务学习模型相比，PLE 的浏览量提高了 2.23%，观看时间提高了 1.84%，这是一个显著的改进，证明了 PLE 的有效性。最后，在公共基准数据集上的大量离线实验表明，PLE 不仅适用于推荐场景，还可以应用于各种其他场景以消除跷跷板现象。PLE 现已成功部署到腾讯的在线视频推荐系统中。
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
![输入图片说明](/imgs/2025-07-08/9pixHClT4knQb3RC.png)
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
### 4.2 专家网络
#### 共享专家代码实现
```Python
share_output=[expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]] # 输入为（batch_size, input_dim），share_experts为layers_num层，每一层有shared_expert_num个全连接层——MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)，最后输出为（batch_size, 1, bottom_mlp_dims[i]）
```
#### 特殊专家代码实现
```Python
task_output=[expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]] # 输入为（batch_size, input_dim），task_experts为layers_num层，每一层有specific_expert_num个全连接层——MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)，最后输出为（batch_size, 1, bottom_mlp_dims[i]）注：特殊专家网络mlp数量为layers_num*task_num*specific_expert_num
```
#### 特殊专家门控代码实现
```Python
gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1) # 每一个任务都有一个对应的门控结果，因此门控网络数量为layers_num*task_num，torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1))
```
### 4.3 Gate加权输出
#### 代码实现
```Python

```

## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMTM5MzkwMywxMjA2Mjc2ODAzLC0xNT
g2Nzc3NTExLDE5MTg4ODk3ODMsMjEzMjQ5NTk2Nyw2MTM4NDIx
OTEsLTE3NTQxMTY3MjMsMTc5NTc1MDIzMCwyMDgwNTYxNjM0XX
0=
-->