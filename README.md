# medVQA
该仓库提供了(我们方法)的代码，这是一个利用思维链和大语言模型在多模态模型上预训练并微调的能够做医学视觉问答的模型。详情请参考我们的论文：(论文)
## Download
https://drive.google.com/drive/folders/1oICfuuRDst6jJkE2jBzTYL0guo7aGd8S?usp=sharing

在这里我们提供了训练所需的数据，包括：

1. PMC-VQA数据集中训练集和测试集的JSON格式数据。
2. PMC-VQA数据集中训练集和测试集用到的图像视觉特征（vision_features/）。
3. 第一阶段训练QCM-LE和第二阶段训练QCMG-A的检查点（checkpoints/）。

## 目录结构
### data
放数据集
### experiments
放实验结果
### model
放模型检查点
### scripts
放运行的脚本
### src
放主要代码

## VQA(Visual Qustion Answering)
