import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from read_data import *
from create_clients import *
from batch_data import *
from build_model import *
from weight_scalling_factor import *
if __name__ == "__main__":
    file1 = r'C:\Users\张立才\PycharmProjects\pythonProject\venv\data\train-images.idx3-ubyte'
    file2 = r'C:\Users\张立才\PycharmProjects\pythonProject\venv\data\train-labels.idx1-ubyte'
    file3 = r'C:\Users\张立才\PycharmProjects\pythonProject\venv\data\t10k-images.idx3-ubyte'
    file4 = r'C:\Users\张立才\PycharmProjects\pythonProject\venv\data\t10k-labels.idx1-ubyte'
    train_data, train_data_head = loadImageSet(file1)
    train_labels, train_labels_head = loadLabelSet(file2)
    test_data,test_data_head = loadImageSet(file3)
    test_labels,test_labels_head=loadLabelSet(file4)


    #为每个客户端对训练数据进行处理和批处理
    clients = create_clients(train_data, train_labels, num_clients=1, initial='client')
    clients_batched=dict()
    for(client_name,data) in clients.items():
        clients_batched[client_name]=batch_data(data)


    #对测试集进行处理和批处理
    test_batched = tf.data.Dataset.from_tensor_slices((test_data,test_labels)).batch(len(test_labels))


    #为模型编译定义一个优化器、损失函数和度量
    lr=0.01
    comms_round=100
    loss='categorical_crossentropy'
    metrics=['accuracy']
    optimizer = SGD(lr=lr,decay=lr/comms_round,momentum=0.9)


    #初始化全局模型
    smlp_global =SimpleMLP()
    global_model=smlp_global.build(784,10)
    costs = []
    accuracy = []
    iteration=[]
    for i in range(comms_round):
        iteration.append(i)
    #启动全局训练循环
    for comms_round in range(comms_round):
        #得到全局模型的权值-将会作为所有本地模型的初始权值
        global_weights=global_model.get_weights()
        #在scalling后初始化列表来收据本地模型权值
        scaled_local_weight_list=list()
        #随机化客户端数据-用关键字
        client_names=list(clients_batched.keys())
        random.shuffle(client_names)
        #通过每一个客户端循环并且创建新的本地模型
        for client in client_names:
            smlp_local=SimpleMLP()
            local_model=smlp_local.build(784,10)
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)
            #给全局模型权重设置本地模型权重
            local_model.set_weights(global_weights)
            #用本地模型训练客户端数据
            local_model.fit(clients_batched[client],epochs=1,verbose=0)
            #得到模型权重然后添加到列表中
            scaling_factor = weight_scalling_factor(clients_batched,client)
            scaled_weights=scale_model_weights(local_model.get_weights(),scaling_factor)
            scaled_local_weight_list.append(scaled_weights)
            #在每轮交流后清除会话以释放内存
            K.clear_session()
        #得到所有本地模型的平均值，我们简单的把scaled权值相加
        average_weights=sum_scaled_weights(scaled_local_weight_list)
        #更新全局模型
        global_model.set_weights(average_weights)

        #在每轮交流后测试全局模型并且输出精度
        for(test_data,test_labels) in test_batched:
            global_acc,global_loss =test_model(test_data,test_labels,global_model,comms_round)
            costs.append(global_loss)
            accuracy.append(global_acc)
    plt.plot(iteration, costs, "bo-", linewidth=2, markersize=12, label="First")
    plt.plot(iteration, accuracy, "gs-", linewidth=2, markersize=12, label="Sceond")
    plt.ylabel('cost')
    plt.title("learning rate={}".format(lr))
    plt.show()