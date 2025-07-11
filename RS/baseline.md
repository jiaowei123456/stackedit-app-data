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

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMTYzOTY4MDAsMzQwNjQ3NzYyLC0yOD
M0MzE5MzQsNDk3ODE4ODEwLDQ0MDkwNTYxOV19
-->