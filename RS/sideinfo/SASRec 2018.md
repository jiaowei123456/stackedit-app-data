# Self-Attentive Sequential Recommendation
[原文链接]()
## 0 摘要：
序列动态是许多现代推荐系统的关键特性，这些系统试图根据用户近期执行的操作来捕捉其活动的“上下文”。为了捕捉此类模式，两种方法已广泛流行：马尔可夫链（MCs）和循环神经网络（RNNs）。马尔可夫链假设用户接下来的动作仅能基于其最近（或最近几次）的动作来预测，而循环神经网络原则上允许发现更长期的语义。一般来说，**基于马尔可夫链的方法在极其稀疏的数据集中表现最佳**，在这种情况下模型的简洁性至关重要，而**循环神经网络在更密集的数据集中表现更好**，在这种情况下可以承受更高的模型复杂度。我们工作的目标是平衡这两个目标，通过**提出一种基于自注意力机制的序列模型（SASRec），它能够像循环神经网络一样捕捉长期语义，但通过注意力机制，仅基于相对较少的动作进行预测，就像马尔可夫链那样**。在每个时间步长，SASRec 都试图从用户的动作历史中识别出哪些项目是“相关的”，并利用它们来预测下一个项目。大量的实证研究表明，我们的方法在稀疏和密集数据集上均优于各种最先进的序列模型（包括基于 MC/CNN/RNN 的方法）。此外，该模型的效率比同类基于 CNN/RNN 的模型高出一个数量级。注意力权重的可视化也展示了我们的模型如何自适应地处理不同密度的数据集，并揭示了活动序列中的有意义模式。

## 背景
序列推荐系统的目标是将基于用户历史行为的个性化模型与某种“情境”概念相结合，同时基于用户最近的操作来考虑这些情境因素。从序列动态中捕捉有用模式是一项具有挑战性的任务，主要是因为**输入空间的维度会随着用于构成情境的过往操作数量的增加而呈指数级增长。因此，序列推荐领域的研究主要集中在如何简洁地捕捉这些高阶动态方面**。
## 1 论文解决的问题：
马尔可夫链（MCs）就是一个典型的例子，它假定下一个动作仅取决于之前的动作（或之前的几项）。这种方法已被成功应用于描述推荐中的短期项目转换。另一项研究则使用循环神经网络（RNNs）通过一个隐藏状态来总结所有之前的动作，该隐藏状态用于预测下一个动作。这两种方法虽然在特定情况下表现良好，但都对某些类型的数据有一定的局限性。基于蒙特卡罗的方法通过做出严格的简化假设，在高稀疏度的环境中表现良好，但在处理更复杂的情况时可能无法捕捉到其复杂的动态变化。相反，循环神经网络虽然具有很强的表达能力，但在能够超越更简单的基准模型之前，需要大量的数据（尤其是密集的数据）。	
## 2 论文创新点：
受到Transformer的启发，我们试图将自注意力机制应用于序列推荐问题中。我们的期望是，这个想法能够解决上述提到的两个问题，**一方面能够从过去的所有操作中获取背景信息（就像循环神经网络那样），另一方面能够仅基于少量的操作来构建预测（就像马尔可夫链那样）**。具体而言，我们构建了一个基于自注意力的序列推荐模型（SASRec），该模型在每个时间步会自适应地为之前的项目分配权重。
所提出的模型在多个基准数据集上的表现明显优于目前最先进的基于 MC/CNN/RNN 的序列推荐方法。特别是，我们研究了模型性能随数据集稀疏程度的变化情况，结果表明模型性能与上述描述的模式高度一致。由于采用了自注意力机制，SASRec 在密集数据集中倾向于考虑长距离依赖关系，而在稀疏数据集中则更侧重于近期活动。这对于灵活处理不同密度的数据集至关重要。
SASRec 的“块”结构适用于并行加速，从而使得该模型的速度比基于 CNN/RNN 的替代方案快一个数量级。此外，我们分析了 SASRec 的复杂性和可扩展性，进行了全面的消融研究以展示关键组件的影响，并通过可视化注意力权重来定量地揭示模型的行为。

### 2.1 模型细节：
模型和transformer没啥区别，细节参考如下：
https://blog.csdn.net/CRW__DREAM/article/details/124115007

### 2.1 技巧：


## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbOTkzMzEwMzQ3LC01NDAxMTk5NDUsMTEzMT
gyOTUwOCwyMzM5OTg3OTcsLTExNDg4ODUzNjddfQ==
-->