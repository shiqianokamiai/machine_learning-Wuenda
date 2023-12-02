Logistic Regression with a Neural Network mindset
-
### Purposes:
1. build the genenral architecture of a learning algorithm
创建一个的总的学习算法结构
-   initializing parameters 
初始化各个参数 如w（权重）和b（偏差）
-  Calculating the cost function and its gradient 
计算梯度时所的损耗
- Using an optimization algorithm(gradient descent) 
运用梯度下降法计算出最优的偏差和权重 
2.Gather all three functions together into a main model, in the right order 
将所有的作用结合在一个模型之内

### Packages:
numpy, h5py, matplotlib, PIL and scipy 还有 作业中自带的代码（模块）lr_utils

### Problem set
**问题集合**包含着两类:
1. 猫(y = 1) 
2. 非猫(y = 0) 
3.长宽比例相同的照片尺寸（num_px, num_px, 3）3为通道数，通道数为1时是黑白灰，通道3为rgb照片


  在**问题集合**之中我们将所有数据后加上_orig来表示还未经过提前处理的数据**(pre-processing)**
 接下来的代码来显示数据集中的一个图片
~~~
index = 5
plt.imshow(train_set_x_orig[index])
pylab.show()
print ("y = " + st······················r(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
~~~
这里有个**注意事项：**原来代码里是没有**pylab.show()**. 当然当import的时候，也要将**pylab**这一模块加入。没有这条是**无法正常显示出图片**的，相关的问题在源代码里也贴上了相应的csdn网址。

当我们想要将一个**矩阵X(a,b,c,d) flatten into X_flatten(b*c*d, a)**。以下代码会有所帮助：
~~~
x_flatten = x.reshape(x.shape[0], -1).T
~~~
将图片数据flatten的目的就是**decreasing memory。**

原文中有关于**sanity check**的代码，中文叫做 合理性检测。用于检测所得出的答案是否合理。

为了表示图片图片内的每个像素大部分情况下都是由**RGB**三色组成。在机器学习中，一个常见的预处理方法就是将数据进行**中心化和标准化（standardize and center）**。当我们处理图片数据，我们最简单和方便的方法就是将每个图片数据除于**255**(每个像素的最大数值)

 -  常见的预处理步骤：
	 1. 判断一个问题的维度和shape
	 2. 将数据集进行重塑
	 3. 标准化数据

![image](https://github.com/shiqianokamiai/machine_learning-Wuenda/assets/151977259/826fff68-a55d-49cb-b309-655c35c6d1ad)
上图就是整个计算的过程。
![image](https://github.com/shiqianokamiai/machine_learning-Wuenda/assets/151977259/2e261de6-23ad-4084-ae8a-dd33bc1718e7)
上图中
	(1)  将flatten data X(i)代入权重和偏差。
    (2) 将z(i)进行sigmoid计算。
    (3)~(6) 损失函数得到最优的权重和偏差。

### Building the parts of our algorithm

1. 定义整个模型的结构，如输入特征的数据
2. 初始化模型的参数
3. 进行循环：
		- 计算出前向传播
		- 计算出后向传播
		- 更新参数
这其中的具体代码就参考 assignment2.py
根据这些公式我们能一步步写出相应的代码内容：	![image](https://github.com/shiqianokamiai/machine_learning-Wuenda/assets/151977259/6e53aeac-67e6-41ea-8b9e-5ec8b47be83d)
![image](https://github.com/shiqianokamiai/machine_learning-Wuenda/assets/151977259/16a2cf51-7606-406d-9f2f-02fec9b48ab7)

最后我们将一切的代码组合后我们会得到一个**cost和iterations**相关的图。

对于这个图我们由一系列的解释：
![Figure_1](https://github.com/shiqianokamiai/machine_learning-Wuenda/assets/151977259/be3af6f0-2e52-4d14-93f9-8e7dce74720d)
- 不同的**学习率**会由不同的**预测结果**和**cost**
- 如果**学习率过大**(0.01)，那么cost将会大幅的地摆动。甚至会出现分歧
- 一个更低的cost并不意味着会有一个更好的模型。我们需要时刻检查是否会**过拟合**。它会发生在**训练准确率过分高于检验准确率**
- 在深度学习中我们会选择一个可以将cost function最小化的学习率
- 如果出现了**过拟合**的现象，我们就要进行一些操作(后续的学习中会有所谈到)
