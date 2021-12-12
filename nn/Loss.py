import numpy as np


class Cross_entropy(object):
    """交叉熵

        实现loss函数的计算（交叉熵）

        参数:
            self.output :   预测值
            self.target :   标签值
            self.error  :   交叉熵
    """
    def __init__(self):
        self.output = None
        self.target = None
        self.error = None

    def get_error(self, output, target):
        """

        :param output   :   预测值
        :param target   :   标签值
        :return         :   交叉熵
        """
        self.output = output
        self.target = target

        # 交叉熵计算
        self.error = -1 * np.sum(self.target * np.log(self.output))
        return self.error

    def get_gradient(self):
        """获得梯度值

        :return :   梯度值
        """
        return -1 * self.target / self.output
