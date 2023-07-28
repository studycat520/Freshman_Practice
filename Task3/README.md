# Task 3
### 参数设置
#全局参数

VOCAB = 10000  # imdb’s vocab_size 即词汇表大小

MAX_LEN = 200      # max length

BATCH_SIZE = 256

***
#LSTM

EMB_SIZE = 128   # embedding size

HID_SIZE = 128   # lstm hidden size

DROPOUT = 0.2

***
#GRU

LAYERS = 1

OUTPUTS = 2

***
LR = 0.001 #learning rate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 10
***
#### LSTM网络acc结果  
<img width="184" alt="LSTM_acc" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/16d6288d-1f91-4516-af60-9cccc3c33a14">


#### GRU网络acc结果  
<img width="177" alt="GRU_acc" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/bef92e61-7fe8-4a96-8ca6-5f9fb4df79c5">


### 两个网络的Acc对比
列表显示两个网络分别在每个类别上的 accuracy 和总体的 accuracy 如下：  
<img width="187" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/e884034a-f9af-47b1-aad6-70cb7c969e1e">



### 两个模型的loss曲线
绘制训练和测试阶段的 loss 随训练过程进行的变化曲线如下：
#### LSTM网络的loss曲线
![LSTM_loss](https://github.com/studycat520/Freshman_Practice/assets/68159381/057ef90b-5f5f-47db-a695-01d4554cd490)


#### GRU网络的loss曲线
![GRU_loss](https://github.com/studycat520/Freshman_Practice/assets/68159381/7a9ae6ee-d92c-4a02-9454-24ec714a193a)

