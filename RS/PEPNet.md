# One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
[原文链接](https://doi.org/10.1145/3383313.3412236)
## 0 摘要：
传统的行业推荐系统通常会使用单一领域的数据来训练模型，并服务于该单一领域。然而，一个大规模的商业平台往往包含多个领域，其推荐系统通常**需要对多个领域进行点击率（CTR）预测**。通常，不同的领域可能会有一些共同的用户群体和项目，而每个领域又可能有其独特的用户群体和项目。此外，即使是同一个用户在不同的领域中也可能有不同的行为。为了充分利用来自不同领域的所有数据，可以训练一个单一的模型来服务于所有领域。然而，单个模型难以捕捉各个领域的特征并很好地服务于所有领域。另一方面，为每个领域单独训练一个模型并不能充分利用来自所有领域的数据。在本文中，我们提出了星型拓扑自适应推荐器（Star Topology Adaptive Recommender, STAR）模型，通过同时利用来自所有领域的数据来训练一个单一模型，捕捉每个领域的特征，并对不同领域的共性进行建模。从本质上讲，**每个领域的网络由两个分解网络组成：一个由所有领域共享的中心网络，以及针对每个领域定制的领域特定网络**。对于每个领域，我们将这两个分解网络结合起来，并通过将共享网络的权重与领域特定网络的权重进行元素相乘的方式生成一个统一的网络，尽管这两个分解网络也可以使用其他函数进行组合，这方面的研究还有待进一步开展。最重要的是，STAR 能够从所有数据中学习共享网络，并根据每个领域的特点调整领域特定参数。来自实际数据的实验结果验证了所提出的 STAR 模型的优越性。自 2020 年末以来，STAR 已被部署在阿里巴巴的展示广告系统中，点击率（CTR）提高了 8.0%，每千次展示收入（RPM）增加了 6.0%。
## 1 论文解决的问题：

## 2 论文创新点：

## 3 相关工作：

## 4 模型结构与实现代码：
![输入图片说明](/imgs/2025-07-09/UtZLPMiTfAmVX2li.png)
### 4.1 门控权重生成器（GNU）实现
通过两个全连接层（relu+sigmoid），包含正则化、超参数gamma，防止过拟合并且提高权重占比，输出特征权重，类似于senet。
```Python
class GateNU:  
    def __init__(self,  
                 hidden_units,  
                 gamma=2.,  
                 l2_reg=0.):  
        assert len(hidden_units) == 2  
        self.hidden_units = hidden_units  
        self.gamma = gamma  
        self.l2_reg = l2_reg # 防止过拟合，提升泛化能力  
  
    def __call__(self, inputs):  
        output = tf.layers.dense(inputs, self.hidden_units[0], activation="relu",  
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)) # 引入非线性能力，增强表达力  
        output = tf.layers.dense(output, self.hidden_units[1], activation="sigmoid", # 输出范围 [0, 1]，可解释为重要性权重或注意力分数  
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))  
        return self.gamma * output # 放大输出值，增强 gate 的影响力度（类似 attention 中的温度系数）
```
### 4.2 嵌入个性化网络（EPNet）代码实现
输出GNU加权后的特征，全连接层在GNU中，**需要注意全连接层的输出维度为人为设置需要与emb的维度相同吗？**（猜测相同）
```Python
class EPNet:  
    def __init__(self,  
                 hidden_units,  
                 l2_reg=0.):  
  
        self.gate_nu = GateNU(hidden_units=hidden_units, l2_reg=l2_reg)  
  
  
    def __call__(self, domain, emb):  
        # domain: 当前任务/场景的领域特征（persona），形状为 [B, D]        # emb: 输入嵌入向量，通常是共享特征或上下文特征，形状为 [B, E]        # 使用 tf.stop_gradient 冻结 emb 的梯度，防止 gate 影响其更新  
        # 输出 gate 权重张量，形状为 [B, E]（和 emb 同维度） 这里形状由hidden_units确定  
        # 使用 gate 权重对原始嵌入进行 element-wise 相乘  
        return self.gate_nu(tf.concat([domain, tf.stop_gradient(emb)], axis=-1)) * emb
```
### 4.3 参数个性化网络（Parameter Personalized Network, PPNet）代码实现
本质是多个场景都有多个全连接层（hidden_units），每个全连接层（hidden_units[i]）的输出都受到GNU的加权，每一场景的输出都加入outputs。其中GNU的输入为input拼接persona参数（每一场景的输入都相同，由反向传播进行区分优化）。
```Python
class PPNet:  
    def __init__(self,  
                 multiples,  
                 hidden_units,  
                 activation,  
                 l2_reg=0.,  
                 **kwargs):  
        self.hidden_units = hidden_units  
        self.l2_reg = l2_reg  
        self.activation = activation  
  
        self.multiples = multiples  
        # 创建多个 GateNU 门控网络，每个门控对应一层 MLP 层。  
        # 每个门控输出维度为 i * multiples，然后会被拆分为 multiples 份，供不同路径使用。  
        self.gate_nu = [GateNU([i*self.multiples, i*self.multiples], l2_reg=self.l2_reg) for i in self.hidden_units]  
  
    def __call__(self, inputs, persona):  
        gate_list = [] # 构建 Gate 权重列表，列表数量为 使用 tf.split(..., self.multiples) 将 gate 权重拆分成多个分支  
        for i in range(len(self.hidden_units)):  
            gate = self.gate_nu[i](tf.concat([persona, tf.stop_gradient(inputs)], axis=-1))    # persona是个人特征  
            gate = tf.split(gate, self.multiples, axis=1)  
            gate_list.append(gate)  
  
        output_list = []  
  
        for n in range(self.multiples):  
            output = inputs  
  
            for i in range(len(self.hidden_units)):  
                fc = tf.layers.dense(output, self.hidden_units[i], activation=self.activation,  
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))  
  
                output = gate_list[i][n] * fc  
  
            output_list.append(output)  
  
        return output_list
```
### 4.4 参数化与嵌入个性化网络（PEPNet）代码实现

```Python
class PEPNet:  
    def __init__(self,  
                 fields: List[Field],  
                 num_tasks: int,  
                 dnn_hidden_units: List[int] = [100, 64],  
                 dnn_activation: Union[str, Callable] = "relu",  
                 dnn_l2_reg: float = 0.,  
                 attention_agg: Type[AttentionBase] = Attention,  
                 gru_hidden_size: int = 1,  
                 attention_hidden_units: List[int] = [80, 40],  
                 attention_activation: Callable = tf.nn.sigmoid,  
                 mode: str = "concat"):  
        self.embedding_table = {}  
        self.num_tasks = num_tasks  
  
        self.epnet = partial(EPNet, l2_reg=dnn_l2_reg)  
        self.ppnet = PPNet(num_tasks, dnn_hidden_units, dnn_activation, dnn_l2_reg)  
  
        with tf.variable_scope(name_or_scope='attention_layer'):  
            self.attention_agg = attention_agg(gru_hidden_size, attention_hidden_units, attention_activation)  
  
    def embedding(self, inputs_dict):  
        result = []  
        for name in inputs_dict:  
            result.append(  
                tf.nn.embedding_lookup(self.embedding_table[name], inputs_dict[name])  
            )  
        return self.func(result)  
  
    def predict_layer(self, inputs):  
        output = tf.layers.dense(inputs, 1, activation=tf.nn.sigmoid,  
                                 kernel_initializer=tf.glorot_normal_initializer())  
        return tf.reshape(output, [-1])  
  
    def __call__(self,  
                 user_behaviors_ids: Dict[str, tf.Tensor],  
                 sequence_length: tf.Tensor,  
                 user_ids: Dict[str, tf.Tensor],  
                 item_ids: Dict[str, tf.Tensor],  
                 other_feature_ids: Dict[str, tf.Tensor],  
                 domain_ids: Dict[str, Dict[str, tf.Tensor]],  
                 ) -> Dict[str, List[tf.Tensor]]:  
        """  
  
        :param user_behaviors_ids: 用户行为序列ID [B, N], 支持多种属性组合，如goods_id+shop_id+cate_id  
        :param sequence_length: 用户行为序列长度 [B]        :param user_ids: 用户个性化特征  
        :param item_ids: 候选items个性化特征  
        :param other_feature_ids: 其他特征，如用户特征及上下文特征  
        :param domain_ids: 每个场景的所有特征，key为场景名称，value如上user_ids和item_ids等  
        :return: 每个场景的所有task预估列表  
        """         
        with tf.variable_scope(name_or_scope='attention_layer'): # 如 DIN 中的 attention 结构  
            att_outputs = self.attention_agg(user_behaviors_embeddings, item_embeddings, sequence_length) # [B, T, H]，[B, H]，[B]  
            if isinstance(att_outputs, (list, tuple)):  
                att_outputs = att_outputs[-1]   # [B, H]  
  
        inputs = tf.concat([att_outputs, other_feature_embeddings], axis=-1)    # 拼接其他特征[B, H + D]  
        inputs_dim = inputs.shape.as_list()[-1]  
        epnet = self.epnet(hidden_units=[inputs_dim, inputs_dim])  
  
        output_dict = {}  
        # compute each domain's prediction  
        for domain in domain_embeddings:  
            ep_emb = epnet(domain_embeddings[domain], inputs) # 获取领域感知加权后的emb [B, H + D]  
  
            pp_outputs = self.ppnet(ep_emb, tf.concat([user_embeddings, item_embeddings], axis=-1)) # PPNet：多路径个性化建模 output_list 是一个长度为 num_task 的列表，每个元素是一个 [B, hidden_units[-1]] 的张量  
  
            # compute each task's prediction in special domain  
            task_outputs = []  
            for i in range(self.num_tasks):  
                task_outputs.append(self.predict_layer(pp_outputs[i]))  
  
            output_dict[domain] = task_outputs  
  
        return output_dict # [B, num_domain, num_task]
```
## 5 实验与分析：

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MTIxNzQwNTYsNjIwNjQ0NTM5LDYyMD
Y0NDUzOSwtMjA2MzUzMzc1MSwtMTA4ODM0MzkzNCwtMTIwMzUz
MjY0NF19
-->