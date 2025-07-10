# Shopee Network code
## 基线wang l
```Python
def attention(queries, keys, keys_length):  
    '''  
        queries:     [B, H]    [batch_size,embedding_size]        keys:        [B, T, H]   [batch_size,T,embedding_size]        keys_length: [B]        [batch_size]        #T为历史行为序列长度  
    '''  
    # (?,32)->(None,32)->32  
    # tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变  
    # tf.shape(keys)[1]==T  
    # 对queries的维度进行reshape  
    # (?,T,32)这里是为了让queries和keys的维度相同而做的操作  
    # (?,T,128)把u和v以及u v的element wise差值向量合并起来作为输入，  
    # 然后喂给全连接层，最后得出两个item embedding，比如u和v的权重，即g(Vi,Va)  
  
    queries_hidden_units = queries.get_shape().as_list()[-1]  
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B*T*4H  
  
    # 三层全链接(d_layer_3_all为训练出来的atteneion权重）  
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')  
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')  
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1  
  
    # 为了让outputs维度和keys的维度一致  
    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T  
  
    #  bool类型 tf.shape(keys)[1]为历史行为序列的最大长度，keys_length为人为设定的参数，  
    #  如tf.sequence_mask(5，3)  即为array[True,True,True,False,False]  
    #  函数的作用是为了后面补齐行为序列，获取等长的行为序列做铺垫  
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  
  
    # 在第二维增加一维，也就是由B*T变成B*1*T  
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T  
  
    # tf.ones_like新建一个与output类型大小一致的tensor，设置填充值为一个很小的值，而不是0,padding的mask后补一个很小的负数，这样softmax之后就会接近0  
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  
  
    # 填充，获取等长的行为序列  
    # tf.where(condition， x, y),condition是bool型值，True/False，返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素  
    # 由于是替换，返回值的维度，和condition，x ， y都是相等的。  
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T  
  
    # Scale（缩放）  
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)  
    # Activation  
    outputs = tf.nn.softmax(outputs)  # B * 1 * T  
    # Weighted Sum outputs=g(Vi,Va)   keys=Vi    # 这步为公式中的g(Vi*Va)*Vi  
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))  
    return outputs
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MDEwNjYzNTAsNDk3ODE4ODEwLDQ0MD
kwNTYxOV19
-->