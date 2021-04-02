import random
def create_clients(train_data,train_labels,num_clients=10,initial='clients'):
    #创建客户端列表
    client_names=['{}_{}'.format(initial,i+1) for i in range(num_clients)]
    #随机化数据
    data=list(zip(train_data,train_labels))
    random.shuffle(data)
    #给每个客户端分配碎片数据
    size=len(data)//num_clients
    shards=[data[i:i+size]for i in range(0,size*num_clients,size)]
    #客户端的编号一定等于碎片数据的编号
    assert(len(shards)==len(client_names))
    return {client_names[i]:shards[i] for i in range(len(client_names))}