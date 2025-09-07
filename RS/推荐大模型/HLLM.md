# HLLM:Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling
[原文链接]([2409.12740](https://arxiv.org/pdf/2409.12740))
## 0 摘要：


## 背景


## 1 论文解决的问题：
尽管这些技术有所进步，但将语言模型与推荐系统相结合在复杂性和有效性方面仍面临显著挑战。
* 其中一个问题是，将用户的使用历史行为序列以文本形式输入到语言模型中会导致输入序列非常长。因此，LLM需要比基于 ID-emb的方法更长的序列来表示相同时间段内的用户行为，而语言模型中的自注意力模块的复杂性会随着序列长度的增加而呈平方级增长。
* 此外，推荐单个项目需要生成多个文本标记，这会导致多次前向传递，从而降低效率。在有效性方面，

## 2 论文创新点：


### 2.1 预训练数据集的构建：


### 2.1 技巧：


## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA3NDYxOTU5XX0=
-->