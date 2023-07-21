# Task 2
### 两个模型最终测试结果
正式训练中 epoch=30，batch size=128，learning rate=0.001，采用 Adam 优化器。最终保留残差连接和去除残差连接的两个模型测试结果分别如下：  
#### 保留残差连接模型结果
<img width="354" alt="Resnet18_withRes_result" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/f741c289-cd8f-453b-8b80-dbd9d5d97a64">

#### 去除残差连接模型结果
<img width="352" alt="Resnet18_noRes_result" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/435cf03f-65f9-4dee-bcbb-da22fb1a19ea">

### 两个模型的Acc对比
列表显示两个模型分别在每个类别上的 accuracy 和总体的 accuracy 如下：

<img width="331" alt="image" src="https://github.com/studycat520/Freshman_Practice/assets/68159381/68a2ef2b-0046-44e0-be99-9daa0e80569e">


### 两个模型的loss和acc曲线
绘制训练和测试阶段的 loss 随训练过程进行的变化曲线如下，顺便也加入了 acc 的变化曲线，并将四条曲线画在了一张图中：
#### 保留残差连接模型的曲线
![Resnet18_withRes_Curve](https://github.com/studycat520/Freshman_Practice/assets/68159381/9c779899-3a24-45df-a803-d6054f7e121b)


#### 去除残差连接模型的曲线
![Resnet18_noRes_Curve](https://github.com/studycat520/Freshman_Practice/assets/68159381/869d4268-7a18-43d7-9cb9-885542e3ba5a)
