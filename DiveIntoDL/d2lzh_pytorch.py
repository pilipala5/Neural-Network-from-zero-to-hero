import matplotlib_inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random


def use_svg_display():
    # display with 'svg'
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    # get batch data
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# Linear layer
def linreg(X, w, b):
    # (num_inputs, feature) @ (feature, output)
    return torch.mm(X, w) + b


# MSE
def squared_loss(y_hat, y):
    # In pytorch, do not '/2'
    loss = (y_hat - y.view(y_hat.shape)) ** 2 / 2
    return loss


# Stochastic gradient descent ( Notes: different with book, do not need to '/batch_size'
def sgd(params, lr):
    for param in params:
        param.data += -lr * param.grad


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle book']
    return [text_labels[i] for i in labels]


def show_fashion_mnist(images, names):  # label is string not integer, images -> (num_imgs, height, width)
    use_svg_display()
    num_img = len(images)
    _, axs = plt.subplots(1, num_img, figsize=(12, 12))
    for ax, img, na in zip(axs, images, names):
        ax.imshow(img)
        ax.axis('off')
        ax.text(14, -5, na, ha='center', fontsize=10)
    plt.show()


def load_data_fashion_mnist(batch_size=256):
    root = './Datasets/FashionMNIST'
    mnist_train = torchvision.datasets.FashionMNIST(root, train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root, train=False, transform=transforms.ToTensor(), download=True)

    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 本函数已保存在d2lzh包中⽅便以后使⽤
def train_ch3(net, train_iter, test_iter, loss, num_epochs,
              batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr)
            else:
                optimizer.step()  # “softmax回归的简洁实现”⼀节将⽤到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) ==
                              y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n,
                 test_acc))


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

