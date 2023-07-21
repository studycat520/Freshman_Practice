# Task 1
### 代码basic版本记录
完成了实践1中的所有基础部分内容。成功训练模型，且40Epoch后测试--Accuracy of the network on the 10000 test images: 59%  
其中basic的参数配置为：batch_size = 4  learning_rate = 0.001  
<img width="600" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/803cca78-78cd-4faa-b169-30174425ee7e">
### 代码版本1记录
加入训练中loss曲线的变化，以及准确率曲线的变化的可视化
使用10Epoch，同basic中的参数测试结果如下：
![Figure_2](https://github.com/studycat520/Freshman_Practice/assets/68159381/c4165d66-786b-4d1e-8f1d-558f83714d4c)
### 超参数实验
不同 Epoch 进行了实验（lr=0.001，batch size=4，优化器为 SGD

<img width="238" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/b1ebf5d7-6500-4590-9631-c539070be62a">

不同 learning rate 下的训练效果，分别将 lr 设为 0.01，0.005，0.001，0.0005，0.0001（Epoch=5，batch size=4，优化器为 SGD ）

<img width="299" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/20d4d6f4-8953-4b2c-9bd4-92d9ec29b717">

不同的 batch size 进行了实验，分别将 batch size 设为 2，4，8，16，128（Epoch=5，learning rate=0.001，优化器为 SGD)

<img width="240" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/d47a1b4b-e217-404e-bd07-ea53f424d8f0">

对 SGD 和 Adam 两个不同优化器的对比（Epoch=5，learning rate=0.001，batch size=4 的情况下）

<img width="159" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/26740404-32f1-4443-84a2-62d35cec6cb7">

比对 benchmark 中排名第一的算法 SpinalNet，对网络结构进行了改进。改用 Spinal-CNN 的网络结构后，采用上面实验中训练效果最好的超参数（Epoch=5，learning rate=0.001，batch size=4，优化器选择 SGD），最终在测试集上的准确率达到了 0.80






