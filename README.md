﻿# DR-Detection

Attemption for kaggle(Diabetic Retinopathy Detection)


# 2019/05/27
## 方法
1. 把resnet改成了Inceptionv3，但是还没有用预训练，之后再加
2. batch-size只能用32了，Inceptionv3网络有点大
3. 增强方式只用了旋转+翻转
## 结果
尝试了不做数据平衡，发现不做数据平衡的时候正确率涨的比较快，到80%了，但不知道为什么kappa的评分挺低的，0.64这个样子，训练集和验证集都是这样，而且训练到80%后就基本不涨了。
观察混淆矩阵发现训练结果里根本没有二分类，猜测应该是数据不平衡的原因，所以正确率到80%以后就上不去了。
做数据平衡，训练集的正确率上升比较慢，正确率但kappa分数会比较高，没看懂kappa的计算机制