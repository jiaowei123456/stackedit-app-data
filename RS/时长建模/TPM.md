# Tree based Progressive Regression Model for Watch-Time Prediction in Short-video Recommendation
[原文链接](https://dl.acm.org/doi/10.1145/3580305.3599919)
## 0 摘要：
准确预测观看时长对于提升视频推荐系统的用户参与度至关重要。为此，观看时长预测框架应满足四个特性：首先，尽管观看时长是连续值，但它也是一个**有序变量**，其值之间的相对顺序反映了用户偏好的差异。因此，预测结果应反映这种有序关系。其次，模型应捕捉视频观看行为之间的**条件依赖关系**。例如，用户必须观看完视频的一半才能看完整个视频。第三，用**点估计来建模观看时长忽略了模型可能给出不确定性较高**的结果这一事实，这可能会导致推荐系统出现不良情况。因此，该框架应能意识到预测的不确定性。第四，现实中的推荐系统存在严重的偏差放大问题，因此期望预测结果不存在偏差放大。如何设计一个能同时解决这四个问题的框架仍是一个未被探索的领域。因此，我们提出了基于**树的渐进回归模型**（TPM）来预测观看时长。具体而言，将观看时长的序数等级引入到 TPM 中，并将问题分解为一系列条件相关的分类任务，这些任务组织成树形结构。通过遍历该树可以生成观看时长的期望值，并且**观看时长预测的方差**被明确地引入到目标函数中，作为不确定性的一种度量。此外，我们说明了**后门调整**可以无缝地融入到 TPM 中，从而减轻偏差放大。在公共数据集上进行了广泛的离线评估，并且 TPM 已在拥有超过 3 亿日活跃用户的快手视频应用中部署。结果表明TPM 的表现优于最先进的方法，确实显著提升了视频观看体验。
## 1 论文解决的问题：

## 2 论文创新点：

## 3 相关工作：

## 4 模型结构与实现代码：
```Python
tpm_cls_list = [outputs['tpm_{}'.format(i)][0] for i in range(C.LEAF_NUMS-1)]  
tpm_cls_preds = tf.keras.layers.Concatenate(name="concat_tmp_node_cls", axis=1)(tpm_cls_list) # [None, LEAF_NUMS - 1]  
print(tpm_cls_preds)  
  
def tpm_core2(leaf_nums, cls_preds):  
    depth = int(math.log2(leaf_nums))  
    path_preds = [1]  
    begin_index = 0  
    for d in range(depth):  
        level_nums = int(math.pow(2, d))  
        temp = []  
        for level_i in range(int(level_nums)):  
            temp.append(path_preds[level_i] * (1-cls_preds[:, begin_index + level_i]))  
            temp.append(path_preds[level_i] * cls_preds[:, begin_index + level_i])  
        begin_index += level_nums  
        path_preds = temp  
    return path_preds  
  
path_preds = tpm_core2(C.LEAF_NUMS, tpm_cls_preds)  
path_preds = [tf.reshape(item, shape=[-1, 1]) for item in path_preds]  
print(len(path_preds), path_preds)  
pred_tensor = tf.keras.layers.Concatenate(name="concat_tpm_path", axis=1)(path_preds)  
print("pred_tensor:", pred_tensor)  
mean_bucket = [(C.BEGIN_buckets[i] + C.END_buckets[i]) / 2 for i in range(C.LEAF_NUMS)]  
print("BEGIN_bucket:", C.BEGIN_buckets)  
print("END_buckets:", C.END_buckets)  
print("MEAN_buckets:", mean_bucket)  
  
bucket_val_tensor = tf.constant(mean_bucket, dtype=tf.float32)  
print("bucket_val_tensor:", bucket_val_tensor)  
pred_reg = tf.reduce_sum(pred_tensor * bucket_val_tensor, axis=-1, keepdims=True)  
print("pred_reg:", pred_reg)  
outputs['tpm_reg'] = [pred_reg, pred_reg]  
return outputs
```
## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbMjU4MDgwNTk2XX0=
-->