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

### 2. 输入及参数设置
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI4MzQzMTkzNCw0OTc4MTg4MTAsNDQwOT
A1NjE5XX0=
-->