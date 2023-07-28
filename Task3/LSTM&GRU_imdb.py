import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#全局参数
VOCAB = 10000  # imdb’s vocab_size 即词汇表大小
MAX_LEN = 200      # max length
BATCH_SIZE = 256

#LSTM
EMB_SIZE = 128   # embedding size
HID_SIZE = 128   # lstm hidden size
DROPOUT = 0.2

#GRU
LAYERS = 1
OUTPUTS = 2

LR = 0.001 #learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 10

# 保存训练/测试中loss和acc，用于绘图
train_loss_dict = []
test_loss_dict = []

#最终分类别的准确度
num_classes = 2
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

#one-hot编码
# 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=VOCAB):
    # 创建一个形状为(len(sequences), dimension) 的零矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # 将 results[i] 的指定索引设为 1
    return results


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab, emb_size, hid_size, dropout):
        super(LSTMModel, self).__init__()
        #参数赋一下初始值
        self.vocab = vocab
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout

        self.Embedding = nn.Embedding(self.vocab, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=True)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        """
        input : [bs, maxlen]
        output: [bs, 2]
        """
        x = self.Embedding(x)  # [bs, ml, emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]

# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, vocab, emb_size, hid_size, layer_dim, output_dim):
        """
        vocab:词典长度 emb_size:词向量 hid_size: GRU神经元个数
        layer_dim: GRU的层数 output_dim:隐藏层输出的维度(分类的数量)
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hid_size  ## GRU神经元个数
        self.layer_dim = layer_dim  ## GRU的层数
        # 对文本进行词向量处理
        self.embedding = nn.Embedding(vocab, emb_size)
        # GRU+全连接层
        self.gru = nn.GRU(emb_size, hid_size, layer_dim,
                          batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hid_size, output_dim)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        r_out, h_n = self.gru(embeds, None)  # None 表示初始的 hidden state 为0
        # 选取最后一个时间点的out输出
        out = self.fc1(r_out[:, -1, :])
        return out


def TrainModel(model, device, train_loader, optimizer, epoch):   # 训练模型
    model.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)  # 得到loss
        loss.backward()
        optimizer.step()

        #loss计算
        train_loss += loss.item()
        # if(batch_idx + 1) % 10 == 0:    # 打印loss
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(x), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
    with torch.no_grad():
        test_loss = 0.0
        for i, (xt, yt) in enumerate(test_loader):
            xt, yt = xt.to(DEVICE), yt.to(DEVICE)
            y_t = model(xt)
            loss_t = criterion(y_t, yt)
            test_loss += loss_t.item()
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    train_loss_dict.append(train_loss)
    test_loss_dict.append(test_loss)
    print('Epoch:{:2d},Train_loss:{:.3f},Test_loss:{:.3f}'.format(epoch, train_loss, test_loss))

def TestModel(model, device, test_loader):    # 测试模型
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

def TestModel_Final(model, device, test_loader):    # 测试模型
    allnum = len(test_loader.dataset)
    model.eval()
    criterion = nn.CrossEntropyLoss()  # 累加loss
    test_loss = 0.0
    acc = 0
    for batch_idx, (text, label) in enumerate(test_loader):
        x, y = text.to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()

        # 分类别的准确度
        c = (pred.eq(y.view_as(pred))).squeeze()
        for i in range(len(y)):
            _label = y[i]
            class_correct[_label] += c[i].item()
            class_total[_label] += 1

    test_loss /= len(test_loader)

    print('\nAll Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, allnum,
        100. * acc / allnum))

    print('\nAccuracy of All : {:.0f}%'.format(100. * acc / allnum))
    classes = ('negative', 'positive')
    # 分类别输出
    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == '__main__':
    # 借助Keras加载imdb数据集

    # #one-hot编码法
    # (train_data, y_train), (test_data, y_test) = imdb.load_data(num_words=10000)
    # x_train = vectorize_sequences(train_data)  # 将训练数据向量化
    # x_test = vectorize_sequences(test_data)  # 将测试数据向量化
    # y_train = np.asarray(y_train).astype('float32')  # 将训练数据标签向量化
    # y_test = np.asarray(y_test).astype('float32')  # 将测试数据向量标签化

    #填充法
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
    print(x_train.shape, x_test.shape)
    # 转化为TensorDataset
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    # 转化为 DataLoader
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_sampler = RandomSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model0 = LSTMModel(VOCAB, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
    model = GRUModel(VOCAB, EMB_SIZE, HID_SIZE, LAYERS, OUTPUTS).to(DEVICE)
    model.to(DEVICE)


    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    PATH = 'output/GRUModel.pth'  # 定义模型保存路径

    print("-----------------开始训练-----------------")
    for epoch in range(1, EPOCH+1):  # 10个epoch
        TrainModel(model, DEVICE, train_loader, optimizer, epoch)
        acc = TestModel(model, DEVICE, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
    print("-----------------训练结束-----------------")

    # 检验保存的模型
    print("----------------最终模型测试------------------")
    #best_model = LSTMModel(VOCAB, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
    best_model = GRUModel(VOCAB, EMB_SIZE, HID_SIZE, LAYERS, OUTPUTS).to(DEVICE)
    best_model.load_state_dict(torch.load(PATH))
    TestModel_Final(best_model, DEVICE, test_loader)

    #全部在一张图版本
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(np.arange(len(train_loss_dict)), train_loss_dict, label="Train Loss")
    plt.plot(np.arange(len(test_loss_dict)), test_loss_dict, label="Test Loss")

    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.title('GRU Train and Test Loss')
    plt.show()

