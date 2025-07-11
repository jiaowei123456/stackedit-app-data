# Shopee Network code
## 基线网络分析
### 1. 输入及参数设置
```Python
class Configure(object):
    # todo 20250710 更换new_slot
    # 输入特征 
    USER_DENSE_SLOTS = []  
    ITEM_DENSE_SLOTS += [] 
    DENSE_SLOTS = USER_DENSE_SLOTS + ITEM_DENSE_SLOTS  
    USER_SPARSE_SLOTS = []  
    ITEM_SPARSE_SLOTS = []
    SPARSE_SLOTS = USER_SPARSE_SLOTS + ITEM_SPARSE_SLOTS  
    # 网络超参数设置
    SEQ_70_SLOTS = []  
    SEQ_70_LENGTH = 70  
    EMB_DIM = 12  
    DIFF_END_VALUE = 8  
    LEAF_NUMS = DIFF_END_VALUE * 2  
    TASKS_LIST = [  
        'staytime_10s', 'staytime_30s', 'staytime_60s', 'like', 'comment', 'share', 'follow'  
    ] + ["tpm_{}".format(i) for i in range(LEAF_NUMS-1)]  
    BEGIN_buckets = [float(i) * 2 for i in range(DIFF_END_VALUE)]  
    END_buckets = [(float(i) + 1) * 2 for i in range(DIFF_END_VALUE)]  
    MAX_STAYTIME = 512  
    max_log_end = int(math.log(MAX_STAYTIME, 2))  
    min_log_begin = int(math.log(DIFF_END_VALUE * 2, 2))  
    log_interval = (max_log_end - min_log_begin) / DIFF_END_VALUE  
    for i in range(DIFF_END_VALUE):  
        log_begin = min_log_begin + log_interval * i  
        BEGIN_buckets.append(int(pow(2, log_begin)))  
        log_end = min_log_begin + log_interval * (i + 1)  
        END_buckets.append(int(pow(2, log_end)))  
    EXPERT_NUM = 2  
C = Configure()
```

### 2. 头数为1的多头自注意力机制
```Python
class MyAttentionV3(tf.keras.layers.Layer):  
  
    def __init__(self, dims, num_heads, seq_len, **kwargs):  
        super(MyAttentionV3, self).__init__(**kwargs)  
  
        self.dims = dims  
        self.num_heads = num_heads  
        self.head_depth = dims // num_heads  
        self.seq_lens = seq_len  
        self.fc_query = tf.keras.layers.Dense(dims, activation=None)  
        self.fc_key = tf.keras.layers.Dense(dims, activation=None)  
        self.fc_value = tf.keras.layers.Dense(dims, activation=None)  
        self.fc_merge_heads = tf.keras.layers.Dense(dims, activation=None)  
  
    def transpose_for_heads(self, input_x, seq_length):  
        input_x = tf.reshape(input_x, shape=[-1, seq_length, self.num_heads, self.head_depth])  
        input_x = tf.transpose(input_x, perm=[0, 2, 1, 3])  
        return input_x  
  
    def call(self, query_input, seq_input=None, mask=None):  
        '''  
        query_input: [batch_size, dims]        seq_input: [batch_size, seq_len, dims]        '''        print("query_input:", query_input)  
        print("seq_input:", seq_input)  
        print("mask:", mask)  
        query_input = tf.expand_dims(query_input, axis=[1])  # [batch_size, 1, dims]  
        query_tensor = self.fc_query(query_input)  
        key_tensor = self.fc_key(seq_input)  
        value_tensor = self.fc_value(seq_input)  
        print(query_tensor)  
        print(key_tensor)  
        print(value_tensor)  
  
        query_heads = self.transpose_for_heads(query_tensor, 1)  # [batch_size, num_heads, 1, head_depth]  
        key_heads = self.transpose_for_heads(key_tensor, self.seq_lens)  # [batch_size, num_heads, seq_len, head_depth]  
        value_heads = self.transpose_for_heads(value_tensor, self.seq_lens)  # [batch_size, num_heads, seq_len, head_depth]  
        print(query_heads)  
        print(key_heads)  
        print(value_heads)  
  
        div = tf.math.sqrt(float(self.head_depth))  
        attention_logits = tf.matmul(query_heads, key_heads, transpose_b=True) / div  # [batch_size, num_heads, 1, seq_len]  
        if mask is not None:  
            # 将mask转换为适合attention的形状 [batch_size, 1, 1, seq_len]            mask = tf.where(mask, tf.ones_like(mask, dtype=tf.float32), tf.zeros_like(mask, dtype=tf.float32))  # 转换bool为float  
            mask = mask[:, tf.newaxis, tf.newaxis, :]  # 扩展维度  
            # 应用到所有头和所有查询位置  
            attention_logits = attention_logits * mask + (1-mask) * -1e9  
  
        attention_score = tf.nn.softmax(attention_logits, axis=-1)  # [batch_size, num_heads, 1, seq_len]  
        attention_score = tf.transpose(attention_score, perm=[0, 1, 3, 2]) # [batch_size, num_heads, seq_len, 1]  
        # values = tf.matmul(attention_score, value_heads)  # [batch_size, num_heads, seq_len, head_depth]        values = value_heads * attention_score # [batch_size, num_heads, seq_len, head_depth]  
        values = tf.transpose(values, perm=[0, 2, 1, 3])  # [batch_size, seq_len, num_heads, head_depth]  
        values = tf.reshape(values, shape=[-1, self.seq_lens, self.dims])  
        final = self.fc_merge_heads(values)  
        return tf.reduce_sum(final, axis=1)
```
### 3. 长序列输入设置
```Python
def get_long_seq_inputs(slots, seq_max_len, dim):  
    seq_inputs_dict = dict()  
    seq_len_dict = dict()  
    for slot in slots:  
        seq_tensor, seq_length = ego.get_slots(name=f"seq_{slot}",  
                                               slots=[int(slot)],  
                                               dims=[(seq_max_len, dim)],  
                                               poolings=[ego.Pooling.TILE_NF],  
                                               feature_type=ego.FeatureType.COMMON)  
        seq_len_dict[slot] = seq_length  
        # seq_length shape:(None, 1)  
        seq_mask = tf.sequence_mask(seq_length, maxlen=seq_max_len)  
        # seq_mask shape:(None, 1, max_Len)  
        seq_mask = tf.squeeze(seq_mask, 1)  
        # seq_mask shape:(None, max_Len)  
        seq_inputs_dict[slot] = (slot, seq_tensor, seq_mask)  
  
    return seq_inputs_dict, seq_len_dict
```
### 3. sparse输入设置
```Python
# 函数 get_sparse_inputs() 的作用是从配置中提取用户和物品的稀疏特征（sparse features），并通过 embedding 层转换为稠密向量，最终返回一个字典，其中键是槽位（slot）编号，值是对应的稠密特征向量。  
def get_sparse_inputs():  
    # ego对同一个slot只支持一种pooling方式，对于既要做tile pooling又要做sum pooling的的slot，请全部用tile pooling，然后自己在图里面做sum pooling  
    user_sparse_slots = []  
    item_sparse_slots = []  
    for slot in C.SPARSE_SLOTS:  
        if slot in C.USER_SPARSE_SLOTS:  
            user_sparse_slots.append(slot)  
        if slot in C.ITEM_SPARSE_SLOTS:  
            item_sparse_slots.append(slot)  
  
    user_slot_dim = [C.EMB_DIM] * len(user_sparse_slots)  
    user_pool_methods = [ego.Pooling.AVG] * len(user_sparse_slots)  
    user_sparse_inputs = ego.get_slots(name='user_sparse_inputs',  
                                       slots=[int(slot) for slot in user_sparse_slots],  
                                       dims=user_slot_dim,  
                                       poolings=user_pool_methods,  
                                       feature_type=ego.FeatureType.COMMON)  
  
    item_slot_dim = [C.EMB_DIM] * len(item_sparse_slots)  
    item_pool_methods = [ego.Pooling.AVG] * len(item_sparse_slots)  
    item_sparse_inputs = ego.get_slots(name='item_sparse_inputs',  
                                       slots=[int(slot) for slot in item_sparse_slots],  
                                       dims=item_slot_dim,  
                                       poolings=item_pool_methods,  
                                       feature_type=ego.FeatureType.UNDEFINED)  
    user_sparse_inputs_dict = {slot: input for slot, input in  
                               zip(user_sparse_slots, list(tf.split(user_sparse_inputs, user_slot_dim, axis=1)))}  
    item_sparse_inputs_dict = {slot: input for slot, input in  
                               zip(item_sparse_slots, list(tf.split(item_sparse_inputs, item_slot_dim, axis=1)))}  
    return dict(user_sparse_inputs_dict, **item_sparse_inputs_dict)
```
### 4. dense特征输入设置
```Python
def get_dense_inputs():  
    dense_inputs = []  
    wide_slots = ['1001', '1002', '1003', '2400']  
    wide_dense_inputs = []  
    log_dense_slots = ['2400', '2407', '2408', '2409', '2410', '2411']  
    log_dense_slots += ['2412', '2413', '2414', '2415', '2416', '2417', '2418']  
# 函数 get_dense_inputs() 的作用是获取并处理用户和物品的稠密特征（dense features），并对某些特定槽位的特征进行对数变换和宽特征提取。  
    for slot in C.USER_DENSE_SLOTS:  
        dense_input = ego.get_dense_feature(name=slot, dim=1, feature_type=ego.FeatureType.COMMON)  
        if slot in log_dense_slots:  
            dense_input = tf.math.log1p(dense_input)  
        dense_inputs.append(dense_input)  
        if slot in wide_slots:  
            wide_dense_inputs.append(dense_input)  
  
    for slot in C.ITEM_DENSE_SLOTS:  
        dense_input = ego.get_dense_feature(name=slot, dim=1, feature_type=ego.FeatureType.UNDEFINED)  
        if slot in log_dense_slots:  
            dense_input = tf.math.log1p(dense_input)  
        dense_inputs.append(dense_input)  
        if slot in wide_slots:  
            wide_dense_inputs.append(dense_input)  
    return dense_inputs, wide_dense_inputs
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzAxMjU0NzksLTEzMTYzOTY4MDAsMzQwNj
Q3NzYyLC0yODM0MzE5MzQsNDk3ODE4ODEwLDQ0MDkwNTYxOV19

-->