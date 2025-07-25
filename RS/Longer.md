# LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders
[原文链接](https://doi.org/10.1145/3383313.3412236)
## 0 摘要：
在工业推荐系统中，对**超长用户行为序列**进行建模对于捕捉长期和短期偏好至关重要。现有的解决方案通常依赖于两阶段检索或间接建模范式，这会导致上下游不一致和计算效率低下。在本文中，我们提出了 LONGER，这是一种用于 GPU 高效推荐系统的长序列优化转换器。LONGER 包含（i）一种全局标记机制，用于在长上下文中稳定注意力，（ii）LONGER 模型采用了轻量级的 InnerTransformers 和混合注意力策略的标记合并模块，以降低二次复杂度，以及一系列工程优化，包括混合精度训练和激活重计算、KV 缓存服务，以及用于统一基于 GPU 的密集和稀疏参数更新的全同步模型训练和提供服务框架。在字节跳动的广告和电子商务服务中，LONGER 在离线指标和在线 A/B 测试中始终优于强大的基线模型，验证了其持续的有效性和工业级扩展规律。目前，LONGER 已在字节跳动的 10 多个重要场景中全面部署，为数十亿用户提供服务。
## 1 论文解决的问题：

## 2 论文创新点：

## 3 相关工作：

## 4 模型结构与实现代码：


## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NDk4OTMxODQsNzI1MDcyNTEyLDEyMD
Y3MjE0NDRdfQ==
-->