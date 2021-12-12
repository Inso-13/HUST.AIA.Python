import numpy as np


class Layer(object):
    """神经网络层基类

        实现了神经网络层的基本、必要功能
        在此基类基础上，实现了Linear Relu Softmax Input Output Flatten层等

        属性:
            self.next_layer     :   该神经网络层的下一层
            self.last_layer     :   该神经网络层的上一层
            self.input_shape    :   该神经网络层的输入矩阵形状
            self.output_shape   :   该神经网络层的输出矩阵形状
            self.trainable      :   该神经网络层是否需要训练
            self.input          :   该神经网络层的输入矩阵
            self.output         :   该神经网络层的输出矩阵
            self.gradient       :   该神经网络层的梯度
    """

    def __init__(self, last_layer, trainable=True):
        """初始化函数

        :param last_layer   :   该神经网络层的上一层
        :param trainable    :   该神经网络层是否需要训练
        """
        self.next_layer = None
        self.last_layer = last_layer
        if self.last_layer is not None:
            self.last_layer.next_layer = self
        if self.last_layer is not None:
            self.input_shape = self.last_layer.output_shape
        else:
            self.input_shape = None
        self.trainable = trainable
        self.output_shape = None
        self.input = None
        self.output = None
        self.gradient = None

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   输入矩阵
        :return         :   无
        """
        pass

    def backward(self, gradient):
        """反向传播函数

        :param gradient :   该神经网络层的梯度
        :return         :   无
        """
        pass


class Linear(Layer):
    """全连接层

        最重要的神经网络层之一，实现了全连接层的功能

        属性:
            self.weights    :   全连接层的连接权重矩阵
            self.bias       :   全连接层的连接常数偏移量
    """

    def __init__(self, last_layer, output_units):
        """初始化函数

        :param last_layer   :   该全连接层的上一层
        :param output_units :   该全连接层的输出神经元数
        """
        super().__init__(last_layer=last_layer, trainable=True)
        self.output_shape = (output_units, 1)

        # 初始化连接矩阵权重
        limit = np.sqrt(6 / (self.input_shape[0] + self.output_shape[0]))
        self.weights = np.random.uniform(-1 * limit, limit, size=(self.input_shape[0], self.output_shape[0]))
        self.bias = np.zeros((self.output_shape[0], 1))

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   输入矩阵
        :return         :   无
        """
        self.input = input_x.copy()

        # 相当于y=wT*x+b
        self.output = self.weights.T.dot(self.input) + self.bias
        self.next_layer.forward(self.output)

    def backward(self, gradient):
        """反向传播函数

        :param gradient :   上一层的梯度值
        :return         :   无
        """
        self.gradient = gradient.copy()

        # 反向传播梯度
        last_layer_gradient = self.weights.dot(self.gradient)
        self.last_layer.backward(gradient=last_layer_gradient)

        # w的梯度系数实际上是输入矩阵x
        # b的梯度系数始终为1
        grad_for_w = np.tile(self.input.T, self.output_shape)

        # 更新参数
        self.weights -= (grad_for_w * self.gradient).T
        self.bias -= self.gradient


class Relu(Layer):
    """非线性化激活层

        负责激活全连接层的输出，使神经网络非线性化

    """

    def __init__(self, last_layer=None):
        """初始化函数

        :param last_layer   : 该激活函数层的上一层
        """
        super().__init__(last_layer=last_layer, trainable=False)
        self.output_shape = self.input_shape

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   输入矩阵
        :return         :   无
        """
        self.input = input_x.copy()
        self.next_layer.forward(input_x=np.maximum(input_x, 0))

    def backward(self, gradient):
        """反向传播函数

        :param gradient :    梯度值
        :return         :    无
        """
        self.gradient = gradient.copy()

        # 根据Relu函数的分段特性知，
        #   如果输入大于0，对应位置的梯度等于上一层回传的梯度，否则为零
        select_mat = np.zeros(shape=self.input.shape)
        select_mat = np.greater(self.input, select_mat).astype(np.int32)
        last_layer_gradient = select_mat * self.gradient
        self.last_layer.backward(gradient=last_layer_gradient)


class Softmax(Layer):
    """Softmax多分类层

        负责将被激活的全连接层的输出结果归一化后计算分类

        属性:
            self.exp_input      :   输入矩阵的exp值
            self.sum_exp_input  :   输入矩阵的exp值的和
            self.tp             :   self.exp_input / self.sum_exp_input
                                    用于计算梯度
    """

    def __init__(self, last_layer=None):
        """初始化函数

        :param last_layer   :   模型的上一层
        """
        super().__init__(last_layer=last_layer, trainable=False)
        self.output_shape = self.input_shape
        self.exp_input = None
        self.sum_exp_input = None
        self.tp = None

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   输入矩阵
        :return         :   无
        """
        self.input = input_x.copy()
        self.exp_input = np.exp(self.input)
        self.sum_exp_input = np.sum(self.exp_input)
        self.output = self.exp_input / self.sum_exp_input
        self.next_layer.forward(input_x=self.output)

    def backward(self, gradient):
        """反向传播函数

        :param gradient     :   上一层的梯度
        :return             :   无
        """
        self.gradient = gradient.copy()
        self.tp = self.exp_input / self.sum_exp_input
        last_layer_gradient = np.zeros(shape=self.input_shape, dtype=np.float64)

        # 梯度计算
        for i in range(self.input_shape[0]):
            gradient_for_Ii = np.zeros(shape=self.input_shape, dtype=np.float64)

            # 分类标签值与预测值是否相等讨论
            for j in range(self.input_shape[0]):
                if i == j:
                    gradient_for_Ii[j] = self.output[i] * (1 - self.output[i])
                else:
                    gradient_for_Ii[j] = -1 * self.output[i] * self.output[j]

            last_layer_gradient[i] = np.sum(gradient_for_Ii * self.gradient)

        self.last_layer.backward(gradient=last_layer_gradient)


class Input(Layer):
    """输入层

        接受起初输入的数据
        一轮训练开始时使用

    """
    def __init__(self, input_shape):
        """初始化函数

        :param input_shape:
        """
        super().__init__(last_layer=None, trainable=False)
        self.output_shape = input_shape

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   输入的数据
        :return         :   无
        """
        self.input = input_x.copy()
        self.next_layer.forward(input_x=input_x)


class Output(Layer):
    """输出层

        输出最终计算得到的数据
        一轮训练结束时使用

    """

    def __init__(self, last_layer=None):
        """初始化函数

        :param last_layer   :   模型上一层
        """
        super().__init__(last_layer=last_layer, trainable=False)

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   输入矩阵
        :return         :   无
        """
        self.output = input_x.copy()

    def backward(self, gradient):
        """反向传播函数

        :param gradient :   上一层的梯度
        :return         :   无
        """
        self.gradient = gradient.copy()
        self.last_layer.backward(gradient=self.gradient)


class Flatten(Layer):
    """线性化层

        例如输入矩阵形状为(a, b, c), 则输出矩阵形状为(a*b*c, 1)

    """

    def __init__(self, last_layer=None):
        """初始化函数

        :param last_layer   :   模型的上一层
        """
        super().__init__(last_layer=last_layer, trainable=False)
        self.output_shape = (np.prod(self.input_shape), 1)

    def forward(self, input_x):
        """前向计算函数

        :param input_x  :   当前层的输入
        :return         :   无
        """
        self.input = input_x.copy()
        self.next_layer.forward(input_x=self.input.reshape(self.output_shape))

    def backward(self, gradient):
        """反向传播函数

        :param gradient :   当前层输出对损失函数的梯度
        :return         :   无
        """
        self.gradient = gradient.copy()
        last_layer_gradient = self.gradient.reshape(self.input_shape)
        self.last_layer.backward(gradient=last_layer_gradient)
