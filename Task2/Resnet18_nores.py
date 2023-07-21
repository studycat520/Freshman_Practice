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
#Basic块
class BasicBlock(torch.nn.Module):
    def __init__(self,In_channel,Out_channel,Downsample=False):
        super(BasicBlock, self).__init__()
        self.stride = 1
        if Downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Out_channel, 3, self.stride, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Out_channel, Out_channel, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
        )

    def forward(self, x):
        return self.layer(x)

#Res18网络部分
class ResNet18(torch.nn.Module):
    def __init__(self, cifar_class=10):
        super(ResNet18, self).__init__()
        self.features = torch.nn.Sequential(
            #conv1
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),

            #conv2_x
            torch.nn.MaxPool2d(3,2,1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),

            # conv3_x
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),

            # conv4_x
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),

            # conv5_x
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512,cifar_class)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,512)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    #chose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    # 封装一组转换函数对象作为转换器
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))# 用均值和标准差归一化张量图像
    ])


    #超参数部分
    #批量大小
    batch_size = 128
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
    # CIFAR10数据集所有类别名称，用于求每个类别的acc
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    # 实例化模型对象
    model = ResNet18()
    model.to(device)

    # loss and optimizer
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    # #SGD
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #这次换Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #训练Epoch
    epochs = 30

    #保存训练/测试中loss和acc，用于绘图
    train_loss_dict = []
    train_acc_dict = []

    eval_loss_dict = []
    eval_acc_dict = []

    # train
    print("-----------------开始训练-----------------")
    for epoch in range(epochs):  # 训练epoch自定
        train_loss, train_acc = 0.0, 0.0

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
            tp_train = (predicted == labels).sum().item()
            acc_train = tp_train / labels.size(0)
            train_acc += acc_train

        train_acc_dict.append(train_acc / len(train_loader))
        train_loss_dict.append(train_loss / len(train_loader))

        print("----------------开始测试------------------")
        with torch.no_grad():
            eval_loss = 0
            eval_acc = 0
            for data in test_loader:
                images, labels = data
                images = images.to('cuda')
                labels = labels.to('cuda')

                # 模型输出预测结果
                outputs_t = model(images)
                loss_t = criterion(outputs_t, labels)
                # 记录误差
                eval_loss += loss_t.item()

                # 选择置信度最高的类别作为预测类别
                _, predicted = torch.max(outputs_t.data, 1)

                # 记录准确率
                num_correct = (predicted == labels).sum().item()  # 判断是否预测正确
                acc = num_correct / labels.size(0)  # 计算准确率
                eval_acc += acc

            eval_loss_dict.append(eval_loss / len(test_loader))
            eval_acc_dict.append(eval_acc / len(test_loader))

        print('Epoch:{:2d},Train_acc:{:.2f}%,Train_loss:{:.3f},Test_acc:{:.2f}%,Test_loss:{:.3f}'.format(epoch + 1, (train_acc / len(train_loader)) * 100, train_loss / len(train_loader),
                                                                                                         (eval_acc / len(test_loader)) * 100, eval_loss / len(test_loader)))


    print("-----------------训练结束-----------------")

    # save
    # 保存模型参数
    PATH = './output/resnet18.pth'  # 指定模型参数保存路径
    torch.save(model.state_dict(), PATH)  # 保存模型参数

    #test
    PATH = './output/resnet18.pth'  # 指定模型参数保存路径
    # 加载模型参数
    model.load_state_dict(torch.load(PATH))
    correct_t = 0  # 预测正确的数量
    total_t = 0  # 测试集的总数

    print("----------------最终模型测试------------------")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')

            # 模型输出预测结果
            outputs_t = model(images)

            # 选择置信度最高的类别作为预测类别
            _, predicted = torch.max(outputs_t.data, 1)
            total_t += labels.size(0)
            correct_t += (predicted == labels).sum().item()

            #分类别的准确度
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                _label = labels[i]
                class_correct[_label] += c[i].item()
                class_total[_label] += 1

    #画训练、测试曲线
    # # （分开版本
    # epochs_range = range(epochs)
    # plt.figure(figsize=(12, 3))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, train_acc_dict, label="Training Accuracy")
    # plt.legend(loc='lower right')
    # plt.plot(epochs_range, eval_acc_dict, label="Test Accuracy")
    # plt.legend(loc='lower right')
    # plt.title('Resnet18 Accuracy')
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, train_loss_dict, label="Training Loss")
    # plt.legend(loc='upper right')
    # plt.plot(epochs_range, eval_loss_dict, label="Test Loss")
    # plt.legend(loc='upper right')
    # plt.title('Resnet18 Loss')
    # plt.show()


    #全部在一张图版本
    plt.plot(np.arange(len(train_loss_dict)), train_loss_dict, label="Train Loss")

    plt.plot(np.arange(len(train_acc_dict)), train_acc_dict, label="Train Acc")

    plt.plot(np.arange(len(eval_loss_dict)), eval_loss_dict, label="Test Loss")

    plt.plot(np.arange(len(eval_acc_dict)), eval_acc_dict, label="Test Acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    # plt.ylabel("epoch")
    plt.title('Resnet18 Accuracy and Loss')
    plt.show()

    # 输出模型的预测准确率
    print('Accuracy of the network on the 10000 test images: ',correct_t / total_t)

    # 分类别输出
    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



