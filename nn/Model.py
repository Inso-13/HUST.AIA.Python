import numpy as np


class Model(object):
    """神经网络模型

        神经网络模型，高层API

        属性:
            self.input_layer  :     模型的输入层
            self.output_layer :     模型的输出层
            self.loss         :     模型的损失函数
            self.opt          :     模型的优化子

    """
    def __init__(self, input_layer=None, output_layer=None, loss=None, opt=None):
        """初始化函数

        :param input_layer  :   模型的输入层
        :param output_layer :   模型的输出层
        :param loss         :   模型的损失函数
        :param opt          :   模型的优化子
        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.loss = loss
        self.opt = opt
        self.opt.set_model(self)

    def fit(self, x_train, y_train, epochs=50, batch_size=1500):
        """模型训练函数

        :param x_train      :   用于训练的输入值
        :param y_train      :   用于训练的标签值
        :param epochs       :   训练的总轮数
        :param batch_size   :   训练的batch大小
        :return             :   训练过程中的loss值

        """

        # 用于跟踪训练过程中的loss值
        E = []

        train_size = x_train.shape[0]

        # 对于训练的每一轮
        for epoch in range(epochs):
            e = 0

            # 随机选择batch_size大小数据集训练
            for i in range(batch_size):
                ridx = np.sum(np.random.randint(0, train_size, (1,)))
                error = self.train_once(input_x=x_train[ridx], target=y_train[ridx])
                e += error
            E.append(e / batch_size)
            print('epoch:', epoch, '  loss:', e / batch_size)
        return E

    def predict(self, input_x):
        """预测函数

        :param input_x  :   需要预测的输入值
        :return         :   模型的输出值

        """
        self.input_layer.forward(input_x=input_x)
        output = self.output_layer.output
        return output

    def train_once(self, input_x, target):
        """

        :param input_x  :   输入数据
        :param target   :   训练目标（标签值）
        :return         :   单次训练的loss值

        """
        self.input_layer.forward(input_x=input_x)
        output = self.output_layer.output

        # 单次训练的loss值
        error = self.loss.get_error(output=output, target=target)
        self.output_layer.gradient = self.loss.get_gradient()

        # 优化子步进，更新参数
        self.opt.step()
        return error
