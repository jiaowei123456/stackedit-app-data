# Shopee Network code
## 基线网络分析
```Python
class Configure(object):  
    # todo 20250710 更换new_slot  
    # scripts generate  
    USER_DENSE_SLOTS = []  
  
    # steamer stat 14d (lg)  
    ITEM_DENSE_SLOTS += [] 

    DENSE_SLOTS = USER_DENSE_SLOTS + ITEM_DENSE_SLOTS  
  
    USER_SPARSE_SLOTS = []  # context feature cross-feature  
  
    ITEM_SPARSE_SLOTS = ['3001', '3002'] + ['3502', '3503', '3505', '3506'] + ['3520', '3521', '3522'] # cvr使用，可加特征  
  
    ITEM_SPARSE_SLOTS += ['4001', '4021', '4023', '4024', '4052', '4053', '4054', '4055', '4057',  
                          '4058', '4059', '4060'] # cvr使用  
    ITEM_SPARSE_SLOTS += ['4401', '4402', '4403'] # cvr使用，可加特征  
    ITEM_SPARSE_SLOTS += ['2770', '2771', '2772', '2773', '2774', '2777', '2778',  
                          '2787', '2788', '2789', '2790', '2791']  # weiwei cross-feature cvr使用  
  
  
    SPARSE_SLOTS = USER_SPARSE_SLOTS + ITEM_SPARSE_SLOTS  
  
    SEQ_70_SLOTS = ['2970', '2971', '2972']  
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
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc2NzkwMzA3Miw0OTc4MTg4MTAsNDQwOT
A1NjE5XX0=
-->