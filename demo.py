import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from nn import Loss
from nn.Layers import Input, Output, Linear, Relu, Softmax, Flatten
from nn.Model import Model
from nn.optimizer import SGD


def draw_test():
    """数据集展示测试

    :return :   无
    """
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    digits = datasets.load_digits()
    X = np.array(digits.images)
    Y = np.array(digits.target)
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(8, 8), cmap="binary")
        ax.text(0.5, 1, str(Y[i]), color="red")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def linear_mnist_test():
    """全连接神经网络实现mnist数据集分类测试

        使用全连接神经网络，对sklearn中的8*8 mnist数据集分类训练

    """

    # 设置随机种子，保证可复现
    np.random.seed(0)

    # 总类数
    n_classes = 10

    # 获得数据集
    digits = datasets.load_digits()
    X = np.array(digits.images)
    Y = np.array(digits.target)

    # 数据集变形预处理
    X = np.expand_dims(X, -1) / 255
    Y = np.expand_dims(np.eye(n_classes)[Y], -1)

    # 分割数据集
    x_train = X[:1500, ]
    y_train = Y[:1500, ]
    x_test = X[1500:, ]
    y_test = Y[1500:, ]

    # 声明构建每一层神经网络
    input_layer = Input(input_shape=(8, 8, 1))
    flatten = Flatten(last_layer=input_layer)
    d1 = Linear(output_units=64, last_layer=flatten)
    r1 = Relu(last_layer=d1)
    d3 = Linear(output_units=64, last_layer=r1)
    r3 = Relu(last_layer=d3)
    d2 = Linear(output_units=10, last_layer=r3)
    r2 = Relu(last_layer=d2)
    sm = Softmax(last_layer=r2)
    output = Output(last_layer=sm)

    # 随机梯度下降优化子
    opt = SGD(lr=0.05, momentum=0.)

    # 交叉熵损失函数
    ce = Loss.Cross_entropy()

    # 模型构建
    model = Model(input_layer=input_layer, output_layer=output, loss=ce, opt=opt)

    # 模型训练
    losses = model.fit(x_train=x_train, y_train=y_train, epochs=50, batch_size=1500)

    # 模型测试
    cnt = 0
    for i in range(x_test.shape[0]):
        o = model.predict(input_x=x_test[i])
        o = o[..., 0]
        t = y_test[i][..., 0]

        pl = np.argmax(o)
        pt = np.argmax(t)
        if pl == pt:
            cnt += 1
    print('accruacy is ', cnt / x_test.shape[0])

    # 绘制loss变化曲线
    plt.xlabel("epochs")
    plt.subplot(3, 5, 3)
    plt.title("losses")
    plt.plot(losses)

    # 绘制前十张图片的测试结果
    for i in range(10):
        pred = model.predict(input_x=x_test[i])
        pred = pred[..., 0]
        t = y_test[i][..., 0]
        pl = np.argmax(pred)
        pt = np.argmax(t)
        plt.subplot(3, 5, i + 6)
        plt.title("{}->{}".format(pt, pl))
        plt.imshow(x_test[i].reshape(8, 8))
    plt.show()


if __name__ == '__main__':
    # 数据集展示测试
    # draw_test()

    # 全连接神经网络实现mnist数据集分类测试
    linear_mnist_test()
