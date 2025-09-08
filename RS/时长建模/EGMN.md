# Multi-Granularity Distribution Modeling for Video Watch Time Prediction via Exponential-Gaussian Mixture Network
[原文链接](https://arxiv.org/pdf/2508.12665)
## 0 摘要：
准确预测观看时长对于提升短视频平台的用户参与度至关重要，尽管这一过程面临着多粒度级别复杂分布特征带来的挑战。通过对真实工业数据的系统分析，我们从分布角度发现了观看时长预测的两个关键挑战：（1）由大量快速跳过行为导致的粗粒度偏斜；（2）由不同用户与视频互动模式引起的细粒度多样性。因此，我们假设观看时长遵循指数-高斯混合（EGM）分布，其中指数和高斯成分分别表征偏斜和多样性。相应地，我们提出了用于 EGM 分布参数化的指数-高斯混合网络（EGMN），它由两个关键模块组成：隐藏表示编码器和混合参数生成器。我们在公共数据集上进行了广泛的离线实验，并在小红书 App 的工业短视频推荐场景中进行了在线 A/B 测试，以验证 EGMN 相对于现有最先进的方法的优越性。值得注意的是，全面的实验结果已经证明，EGMN 在从粗粒度到细粒度的各个层级上都展现出了出色的分布拟合能力。

## 背景

![输入图片说明](/imgs/2025-09-08/3rSuZYswSklIvveE.png)

## 1 论文解决的问题：


## 2 论文创新点：


### 2.1 预训练数据集的构建：


### 2.1 技巧：


## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk5MDQwNzMxNCwzMDgwNDEwMDRdfQ==
-->