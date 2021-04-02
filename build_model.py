from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
#创建两层MLP作为分类任务的模型
class SimpleMLP:
    @staticmethod
    def build(shape,classes):
        model = Sequential()
        model.add(Dense(200,input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model