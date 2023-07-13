import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#train
# 定义网络结构
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    # 前向传播，加relu
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # x沿着水平方向展开
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    #chose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    # 封装一组转换函数对象作为转换器
    transform = transforms.Compose(  # Compose是transforms的组合类
        [transforms.ToTensor(),  # ToTensor()类把PIL Image格式的图片和Numpy数组转换成张量
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 用均值和标准差归一化张量图像
    )

    #超参数部分
    #批量大小
    batch_size = 4
    #学习率
    learning_rate = 0.001


    # 实例化训练集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=False)
    # 实例化训练集加载器
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=12)
    # 实例化测试集
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transform, download=False)
    # 实例化测试集加载器
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True, num_workers=12)
    # CIFAR10数据集所有类别名称
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 实例化模型对象
    model = Model()
    model.to(device)

    # loss and optimizer
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    #训练Epoch
    epochs = 10
    train_loss_dict = []
    train_acc_dict = []

    # train
    print("-----------------开始训练-----------------")
    for epoch in range(epochs):  # 训练epoch自定
        sizea = len(train_loader.dataset)
        sizel = len(train_loader)

        train_loss, train_acc = 0.0, 0.0
        correct0 = 0  # 预测正确的数量
        total0 = 0  # 训练集的总数

        for i, data in enumerate(train_loader, 0):
            # 获取模型输入；data是由[inputs, labels]组成的列表
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播+反向传播+更新权重
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #计算loss和acc
            #train_acc += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            train_loss += loss.item()
            # 选择置信度最高的类别作为预测类别
            _, predicted = torch.max(outputs.data, 1)
            total0 += labels.size(0)
            correct0 += (predicted == labels).sum().item()

        train_acc = correct0/total0
        train_loss /= sizel
        train_acc_dict.append(train_acc)
        train_loss_dict.append(train_loss)

        print('Epoch:{:2d},Train_acc:{:.1f}%,Train_loss:{:.3f}'.format(epoch + 1, train_acc * 100, train_loss))


    print("-----------------训练结束-----------------")

    #画训练曲线
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_dict, label="Training Accuracy")
    plt.legend(loc='lower right')
    plt.title('Train Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_dict, label="Training Loss")
    plt.legend(loc='upper right')
    plt.title('Train Loss')

    plt.show()

    # save
    # 保存模型参数
    PATH = './output/cifar_model.pth'  # 指定模型参数保存路径
    torch.save(model.state_dict(), PATH)  # 保存模型参数

    #test
    PATH = './output/cifar_model.pth'  # 指定模型参数保存路径
    # 加载模型参数
    model.load_state_dict(torch.load(PATH))
    correct = 0  # 预测正确的数量
    total = 0  # 测试集的总数

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')

            # 模型输出预测结果
            outputs = model(images)

            # 选择置信度最高的类别作为预测类别
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 输出模型的预测准确率
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')


