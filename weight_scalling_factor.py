import tensorflow as tf
from sklearn.metrics import accuracy_score


def weight_scalling_factor(clients_trn_data,client_name):
    client_names=list(clients_trn_data.keys())
    #得到bs
    bs=list(clients_trn_data[client_name])[0][0].shape[0]
    #首先通过客户端计算总训练数据点数
    global_count=sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    #得到一个客户端持有的数据总数
    local_count =tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count

def scale_model_weights(weight,scalar):
    weight_final=[]
    setps=len(weight)
    for i in range(setps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    avg_grad=list()
    #从所有客户端的梯度中得到平均梯度
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple,axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def test_model(X_test,Y_test,model,comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits=model.predict(X_test,batch_size=100)
    logits=model.predict(X_test)
    loss = cce(Y_test,logits)
    acc=accuracy_score(tf.argmax(logits,axis=1),tf.argmax(Y_test,axis=1))
    print('comm_round:{}|global_acc{:.3%}|global_loss:{}'.format(comm_round,acc,loss))
    return acc,loss
