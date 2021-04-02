import  tensorflow as tf
def batch_data(data_shard,bs=32):
    #分割数据碎片为数据和标签
    data,label=zip(*data_shard)
    #处理为tensorflow数据集并进行批处理
    dataset=tf.data.Dataset.from_tensor_slices((list(data),list(label)))
    return dataset.shuffle(len(label)).batch(bs)
