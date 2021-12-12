class SGD(object):
    """随机梯度下降法

        参数:
            self.model      :   该优化子的目标优化模型
            self.lr         :   该优化子的学习率
            self.momentum   :   随机梯度下降法的动量系数
            self.mt         :   当前的累计动量值
    """
    def __init__(self, lr=0.05, momentum=0.0):
        """初始化函数

        :param lr       :   学习率
        :param momentum :   动量系数
        """
        self.model = None
        self.lr = lr
        self.momentum = momentum
        self.mt = 0

    def set_model(self, model):
        """设置该优化子的目标优化模型

        :param model    :   该优化子的目标优化模型
        :return         :   无
        """
        self.model = model

    def step(self):
        """优化子步进一步（开启反向传播优化）

        :return :   无
        """
        self.mt = self.momentum * self.mt + self.lr * self.model.output_layer.gradient
        self.model.output_layer.backward(self.mt)
