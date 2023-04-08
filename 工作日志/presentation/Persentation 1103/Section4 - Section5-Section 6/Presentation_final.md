---
marp: true
---
# <!-- fit --> 机器学习的概念及应用

---
## 机器学习的基本概念
机器学习最直接的解释就是：Giving Computers the Ability to Learn from Data.
计算机，学习算法，数据是机器学习的重要组成。
机器学习的核心是“使用算法解析数据，从中学习，然后对新数据做出决定或预测”。也就是说计算机利用已获取的数据得出某一模型，然后利用此模型进行预测的一种方法，这个过程跟人的学习过程有些类似，比如人获取一定的经验，可以对新问题进行预测。

---
## 机器学习的应用
### 其他领域
机器学习不仅在计算机科学研究中变得越来越重要，而且它也在我们的日常生活中扮演着越来越重要的角色。由于机器学习，我们享受了强大的垃圾邮件过滤器、便捷的文本和语音识别软件、可靠的网络搜索引擎， 关于观看娱乐电影、移动支票存款、预计送餐的建议，以及更多。

---

### 金融领域
随着计算资源和更大数据集不断增加的趋势，机器学习已经成为金融业的一项重要技能。
适用于金融的机器学习操作包括用于资产定价的回归、用于投资组合优化的分类、用于投资组合风险分析和股票选择的聚类、用于市场机制识别的生成模型、用于欺诈检测的特征提取、用于算法交易的强化学习以及用于风险评估、金融预测。

---

**投资组合管理和优化**为了更有效的管理客户们的投资组合，基于机器学习的投资组合管理在当今变得越来越流行。
**欺诈防范和检测**银行和金融机构的欺诈行为会对所有的行业带来深远的负面影响，且恢复欺诈活动的成本将远高于欺诈活动造成的损失。机器学习复杂的算法可以准确发现并确定欺诈模式从而阻止欺诈活动的发生。
**算法交易**算法交易是在没有人工干预的情况下通过算法程序进行订单管理的一种交易模式。交易员会对算法设置一系列的参数（例如交易价格，交易量，和具体的交易时间）。然后，计算机会基于前述设定的算法自动化的下单交易。它通过自动化简化了决策过程从而节省了时间。

---

## 机器学习的三种模式

### 数据

要了解机器学习的模式首先需要了解数据类型。

在数据集中一般：一行数据称为一个**样本**，一列数据称为一个**特征**。有些数据有**目标值（标签值）**，有些数据没有目标值

- 数据类型一：特征值+目标值（目标值是连续的和离散的）
- 数据类型二：只有特征值，没有目标值

---

### 监督式学习

![image-20221004102143833 h:250px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221004102143833.png "image-20221004102143833")

监督学习的主要目标是从带标签的训练数据中学习模型，使我们能够对未知或未来的数据进行预测。在这里**监督**一词指的是已经知道训练样本（输入数据）中期待的输出信号（标签).

---

监督学习过程
![bg h:400px center](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104641187.png)


---

#### 分类

分类是监督学习的一个子类，根据训练集推断它所对应的类别（如：+1，-1），是一种定性输出，也叫离散变量预测。其目标是预测分类类别基于过去观察的新实例或数据点的标签。

例如：根据肿瘤特征判断良性还是恶性，得到的是结果是“良性”或者“恶性”，是离散的。

![w:400px ](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104715361.png)

---

#### 回归

回归也是监督学习的一个子类，在回归分析中，回归分析包括一些预测（**解释**）变量和一个连续的响应变量（**结果**），试图寻找那些变量之间的关系，从而能够让我们预测结果。根据训练集推断它所对应的输出值（实数）是多少，是一种定量输出，也叫连续变量预测。

---
举个例子，我们给一个房价数据集，假设价格与面积楼层有关。在数据集中的每一个样本，我们都给出正确的价格，即这个房子的实际卖价。通过对房子大小楼层等数据进行分析学习，我们可以预测下一个房子的价格。

![image-20221007104730109](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104730109.png)

---

### 强化学习
![image-20221005212900071 h:250px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221005212900071.png "image-20221005212900071")

强化学习的目标是开发一个系统（**智能体**），通过与环境的交互来提高其性能。当前环境状态的信息通常包含所谓的**奖励信号**，可以把强化学习看作一个与监督学习相关的领域。

---

强化学习的反馈并非标定过的正确标签或数值，而是奖励函数对行动度量的结果。**一般模式是强化学习智能体试图通过与环境的一系列交互来最大化奖励。** 每种状态都可以与正或负的奖励相关联，奖励可以被定义为完成一个总目标，如赢棋或输棋。

![image-20221007104741671 h:250px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104741671.png)

强化学习的示例:AlphaGo。


---
### 非监督式学习

![image-20221005142137069 h:250px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221005142137069.png)
在无监督学习中，我们处理的是未标记的数据或未知结构的数据。使用无监督学习技术，我们能够探索在没有已知结果指导的情况下提取有意义信息的数据结构变量或函数。

---
#### 聚类	

**聚类**是探索性的数据分析技术，可以在事先不了解成员关系的情况下，将信息分成有意义的子群（**集群**）。简单说就是一种自动分类的方法，在监督学习中，你很清楚每一个分类是什么，但是聚类则不是，你并不清楚聚类后的几个分类每个代表什么意思。

![image-20221007104755488 h:450px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104755488.png)

---

#### 降维

无监督学习的另一个子领域是降维。通常，我们与 高维度数据——每次观察都伴随着大量的测量——这些数据 会对有限的存储空间和机器的计算性能提出挑战 学习算法。无监督降维是一种常用的特征提取方法 预处理以消除数据中的噪声，噪声会降低某些预测性能 算法。降维将数据压缩到一个更小的维度子空间 保留大部分相关信息。

![image-20221007104804756 h:450px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104804756.png)



---
假设有下图所示的数据集，其中有一系列的样本，它们是 3D 空间中的点，所以现在的样本是三维的，这些数据其实大概都分布在一个平面上。所以这时降维的方法就是**把所有数据都投影到一个二维平面上**
![image-20221007104815718 h:450px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104815718.png)![image-20221007104846223 h:400px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104846223.png)


---

## 机器学习的基本术语与符号

![image-20221007104905361](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221007104905361.png)

---
### 符号
在机器学习中我们将使用矩阵符号来引用我们的数据。同时遵循通用约定，将每个实例表示为特征矩阵X中的单独一行， 其中每个特征存储为单独的列。

由a个例子和b个特征组成的数据集可以写成a×b的矩阵,记作：
$$
X\in\mathbb{R}^{a \times b}
$$

我们将使用上标i来指代第i个训练示例，下标j表示训练的第j个维度数据集。对于鸢尾花实例来说，$x_{1}^{(150)}$指的是花实例150的第一维，即萼片长度。

---

### 术语
**模型**
模型这一词语将会贯穿整个教程的始末，它是机器学习中的核心概念。你可以向它输入数），它就会帮你输出预测结果。整个机器学习的过程都将围绕模型展开，训练出一个优质模型，它可以尽量精准的输出预测值，这就是机器学习的目标。

#### 数据集

数据集，从字面意思很容易理解，它表示一个承载数据的集合，如果缺少了数据集，那么模型就没有存在的意义了。数据集可划分为“训练集”和“测试集”，它们分别在机器学习的“训练阶段”和“预测输出阶段”起着重要的作用。

#### 训练

模型拟合的过程，通过已知的数据和目标，调节算法的参数，最终得到符合预期的模型。

---
#### 样本&特征

样本指的是数据集中的数据，一条数据被称为“一个样本”，通常情况下，样本会包含多个特征值用来描述数据。

#### 假设函数

假设函数可表述为`y=f(x)`其中 x 表示输入数据，而 y 表示输出的预测结果，而这个结果需要不断的优化才会达到预期的结果，否则会与实际值偏差较大。

####  损失函数

损失函数又叫目标函数，简写为 L(x)，这里的 x 是假设函数得出的预测结果“y”，如果 L(x) 的返回值越大就表示预测结果与实际偏差越大，越小则证明预测值越来越“逼近”真实值，这才是机器学习最终的目的。

---

## 机器学习的路线图

### 预处理–将数据整理成形
原始数据很少出现 学习算法的最佳性能所必需的形式和形状。因此， 数据的预处理是任何机器学习应用中最关键的步骤之一。以鸢尾花数据集为例，我们可以想到原始数据 作为我们想要从中提取有意义的特征的一系列花图像。有用的特征可以 以花的颜色或花的高度、长度和宽度为中心。
一些选择的特征可能高度相关，因此在一定程度上是冗余的。 在这些情况下，降维技术对于将特征压缩到 低维子空间。降低我们的特征空间的维数具有以下优点 需要更少的存储空间，并且学习算法可以运行得更快。

---

###  训练和选择预测模型

许多不同的机器学习算法已经被开发出来 来解决不同的问题任务,因此在实践中，至关重要的是 比较至少几个不同的学习算法，以便训练和选择最佳算法 表演模特。我们比较不同的模型之前，我们首先要确定一个度量标准来衡量表现。一个常用的度量是分类准确度，其定义为正确分类实例的比例。

###  评估模型

在我们选择了适合训练数据集的模型之后，我们可以使用测试数据集 来估计它在这个看不见的数据上的表现如何，来估计所谓的泛化误差。

###  使用Python进行机器学习

Python是数据科学领域最流行的编程语言之一，由于它非常活跃的开发者和开源社区，已经开发了大量用于科学计算和机器学习的可用库。

---
# <!-- fit -->Artificial neurons(人工神经元)

---
## 人工神经元
在我们更详细地讨论感知器和相关算法之前，让我们简单地浏览一下机器学习的开端。沃伦·麦卡洛克和沃尔特·皮茨试图理解生物大脑是如何工作的，以便设计人工智能(AI)，他们发表了第一个概念 一种简化的脑细胞，即所谓的麦卡洛克-皮茨(MCP)神经元，在1943年 神经活动中的内在思想，麦卡洛克和皮茨，数学生物物理学通报，生物神经元是大脑中相互连接的神经细胞化学和电子信号的传输。
![fd1b0ebc6d464454e714e51f6da342d](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/fd1b0ebc6d464454e714e51f6da342d.png)

---

麦卡洛克和皮茨将这种神经细胞描述为具有二进制输出的简单逻辑门;多个信号到达树突，然后整合到细胞体，如果积累信号超过某个阈值，就会产生一个输出信号，由轴突传递.仅仅几年后，弗兰克·罗森布拉特发表了感知机学习规则的第一个概念基于MCP神经元模型(由F. Rosenblatt， 康奈尔航空实验室，1957年)。根据他的感知器规则，罗森布拉特提出了一种算法 它会自动学习最佳权重系数，然后乘以输入特征，以便决定神经元是否触发(传输信号)。 在监督学习和分类的背景下，这种算法可以用于预测无论新的数据点属于一个类别还是另一个类别。

---

## 基本的人工神经元的组成
输入 ：输入是我们需要为其预测输出值的一组值。 可以将它们视为数据集中的要素或属性。
权重： 权重是与每个要素关联的实际值，表明了该要素在预测最终值中的重要性
偏差： 偏差用于将激活函数向左或向右移动，在线性方程式中可以称为y截距。
求和函数：求和函数的作用是将权重和输入绑定在一起，以求和。
激活函数：用于在模型中引入非线性。

---

## 对于人工神经元更加正式的定义
更正式地说，我们可以将人工神经元背后的思想放入具有两个类别的二元分类任务中。然后我们可以定义一个决策函数$\sigma$(z) , 采取线性某些输入值x和相应的权重向量w的组合，
$z=w^{1}x^{1}+w^{2}x^{2}+...+w^{n}x^{n}$，其中z是所谓的净输入，如果$\sigma$(z)大于等于特定的阈值$\theta$就把它归为1类，小于特定的阈值就归为0类。$\sigma(z)= \begin{cases} 1,& \text{if z} \geq\theta \\ 0, & \text{otherwise} \end{cases}$  。
为了简化后面的代码实现，我们可以通过几个步骤来修改这个公式。首先我们将阈值$\theta$移到式子的左边得到${z-}\theta\geq{0}$,然后我们定义一个偏差量b，并让${b=-}\theta$,我们就可以得到新的净输入以及新的判断方程。$z=w^{1}x^{1}+w^{2}x^{2}+...+w^{n}x^{n}+b$，$\sigma(z)= \begin{cases} 1,& \text{if z} \geq{0} \\ 0, & \text{otherwise} \end{cases}$ 。

---

## 线性代数基础:点积和矩阵转置
小写黑体为向量，大写黑体为矩阵。矩阵和向量的关系：单个向量可以视为一阶矩阵，多个向量组合在一起就成为多阶矩阵。
$$ 
\pmb a =
{\left[ 
\begin{matrix}
a_{1} \\
a_{2} \\
a_{3}
\end{matrix}
\right]}
$$
$$ 
\pmb b =
{\left[ 
\begin{matrix}
b_{1} \\
b_{2} \\
b_{3}
\end{matrix}
\right]}
$$
上标T代表转置，这是一种运算可以将行列互换。
$$ 
\pmb a^{T} =
{\left[ 
\begin{matrix}
a_{1}a_{2}a_{3}
\end{matrix}
\right]}
$$

---
矩阵相乘：前面矩阵的列必须等于后面矩阵的行。
                                                        $\pmb a^{T}\pmb b=$ $a_{1} \cdot b_{1}+a_{2} \cdot b_{2}+a_{3} \cdot b_{3}$
$$ 
\pmb b\pmb a^{T}=
{\left[ 
\begin{matrix}
b_{1}a_{1}& b_{1}a_{2} &b_{1}a_{3} \\
b_{2}a_{1}& b_{2}a_{2} &b_{2}a_{3}\\
b_{3}a_{1}& b_{3}a_{2} &b_{3}a_{3}
\end{matrix}
\right]}
$$

---
## 为二元分类产生线性判定边界的阈值函数
图2.2说明了净输入$z = w^{T}x + b$被压缩成二进制输出(0或1) 感知器的决策功能(左图)以及如何使用它来区分由线性决策边界分开的两个类别(右图)。
![a6b69bf0b8a36904ceb6d0192acc998](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/a6b69bf0b8a36904ceb6d0192acc998.png)

---
在两个分类中，我们只需要一个线性判别函数$z = w^{T}x + b$。特征空间中所有满足z = 0 的点组成用一个分割超平面（hyperplane），称为决策边界（decision boundary）或决策平面（decision surface）。决策边界将特征空间一分为二，划分成两个区域，每个区域对应一个类别。
超平面：超平面就是三维空间中的平面在更高维空间的推广。d维空间中的超平面是d − 1 维的。在二维空间中，决策边界为一个直线；在三维空间中，决策边界为一个平面；在高维空间中，决策边界为一个超平面。

---

## 感知机学习规则
感知机(perceptron)是二类分类的线性分类模型，其输入为实例的特征向量，输出为实例的类别。
MCP神经元和Rosenblatt的阈值感知器模型背后的整个想法是使用简化方法来模拟大脑中单个神经元的工作方式:它要么激活，要么不激活。因此，罗森布拉特的经典感知器规则相当简单，感知器算法可以通过以下步骤总结：
1.将权重和偏差单元初始化为0或小随机数
2.对于每个训练示例$x^{(i)}$ ：
2.1：计算输出值$\hat{y}^{(i)}$
2..2：更新权重和偏差量

---

## 感知机的概念
![49f1f4088d6b3426ade4f1c6c92c5ee](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/49f1f4088d6b3426ade4f1c6c92c5ee.png)

上图说明了感知机如何接收示例(x)的输入，并将其与偏差量(b)和权重(w)相结合来计算净输入。然后传递净输入到阈值函数生成二进制输出0或1（预测的类标签），在学习阶段，该输出需要与实际数据的类标签进行对比，如果预测错误就需要更新权重和偏差量。

---


## 权重和偏差量的更新
权重和偏差量的更新是感知机的重要部分，在学习阶段需要不断更新权重和偏差量感知机才能收敛。所谓感知机算法的收敛性，就是说只要给定而分类问题是线性可分的，那么按上述感知机算法进行操作，在有限次迭代后一定可以找到一个可将样本完全分类正确的超平面。
更新值计算如下：
                                                            $w_{j}=w_{j}+\Delta w_{j}$
                                                            ${b}={b}+\Delta {b}$
 更新值为原本的值加上新增的值。
 更新值(“增量”)计算如下:         
$$
    \Delta w_{j} = \eta \left(
        y^{(i)} - \hat y^{(i)}
        \right) x_j^{(i)} 
$$
$$
    \Delta b = \eta \left(
        y^{(i)} - \hat y^{(i)}
        \right)
$$

---
  先说偏差量的增量，偏差量的增量等于eta(学习速率，一个常数，可以控制权重更新数值的大小。学习率越低，损失函数的变化速度就越慢，容易过拟合。而学习率过高容易发生梯度爆炸，loss振动幅度较大，模型难以收敛。)乘以原本数据类标签与预测数据类标签的差,不同于偏差量，新增的权重还得乘以对应的特征数据x。

---

  为了方便理解，下面举几个特定的实列来看下权重和偏差量的更新情况。
![577ebc858b0e4c19bf66ed5f23f3e45](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/577ebc858b0e4c19bf66ed5f23f3e45.png)
  例1和例2属于同一类，当预测的类标签与实际的类标签相同时，说明预测对了，权重和偏差量的增值都为0，这时权重和偏差量不改变。
![872a82d9ffb677470cecbef9878f518](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/872a82d9ffb677470cecbef9878f518.png)
 为了更好的理解 让我们看另一个简单的例子。
 当 $y^{(i)}=1,\hat y^{(i)}=0,\eta=1$时，我们假定$x_j^{(i)}$=1.5而且我们把这个例子误归类为0类，在这种情况下，我们想增加权重和偏差量，使预测类标签为1。
 $\Delta w_{j}$=(1-0)1.5=1.5,   $\Delta b$=(1-0)=1。

---


## 线性可分和线性不可分
![ff902e855b50aa1b5a742784d2a7aa2 w:900px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/ff902e855b50aa1b5a742784d2a7aa2.png)
重要的是要注意，感知器的收敛性只有在两个类是线性可分的。图2.3 展示了线性可分和线性不可分场景的可视化示例。如果这两个类不能被一个线性决策边界分开，我们可以设置一个最大值通过训练数据集(时期)的次数和/或容许的错误分类数的阈值，否则感知器将永远不会停止更新权重。在本章的后面，我们将介绍Adaline算法，该算法产生线性决策边界并收敛，即使这些类不是完全线性可分的。在第三章中，我们将学习能够产生非线性决策边界。

---

## 用Python实现感知机学习算法
我们将采用面向对象的方法将感知器接口定义为Python类 将允许我们初始化新的感知器对象，这些对象可以通过fit方法从数据中学习 通过单独的预测方法进行预测。按照惯例，我们给属性添加一个下划线(_) 不是在对象初始化时创建的，但我们通过调用对象的其他 方法，比如self.w_。

```python
import numpy as np
class Perceptron:
 """Perceptron classifier.
 Parameters

 eta : float
 Learning rate (between 0.0 and 1.0)
 n_iter : int
 Passes over the training dataset.
 random_state : int
 Random number generator seed for random weight
 initialization.
 ```

 ---
 ```python
 Attributes

 w_ : 1d-array
 Weights after fitting.
 b_ : Scalar
 Bias unit after fitting.
 errors_ : list
 Number of misclassifications (updates) in each epoch.

 """

 def __init__(self, eta=0.01, n_iter=50, random_state=1):
 self.eta = eta
 self.n_iter = n_iter
 self.random_state = random_state
 ```

 ---
 ```python
def fit(self, X, y):
 """Fit training data.
 Parameters

 X : {array-like}, shape = [n_examples, n_features]
 Training vectors, where n_examples is the number of
 examples and n_features is the number of features.
 y : array-like, shape = [n_examples]
 Target values.

 Returns

 self : object

 """
 rgen = np.random.RandomState(self.random_state)
 self.w_ = rgen.normal(loc=0.0, scale=0.01,
 size=X.shape[1])
 self.b_ = np.float_(0.)
 self.errors_ = []
```

---
```python
 for _ in range(self.n_iter):
 errors = 0
 for xi, target in zip(X, y):
 update = self.eta * (target - self.predict(xi))
 self.w_ += update * xi
 self.b_ += update
 errors += int(update != 0.0)
 self.errors_.append(errors)
 return self

 def net_input(self, X):
 """Calculate net input"""
 return np.dot(X, self.w_) + self.b_

 def predict(self, X):
 """Return class label after unit step"""
 return np.where(self.net_input(X) >= 0.0, 1, 0)
```

---

使用这种感知器实现，我们可以通过初始化$\eta$(学习速率)以及历元数n_iter(通过训练数据集),来初始化新的感知器对象。通过拟合方法，我们将偏差self.b_初始化为初始值0，并将self.w_中的权重初始化为 向量$\mathbb{R}^{m}$，其中m代表数据集中的维数。请注意，初始权重向量包含来自正态分布的小随机数 通过 rgen.normal(loc=0.0，scale=0.01，size=1 + X.shape[1])，其中rgen是一个NumPy 随机数生成器，我们用用户指定的随机种子进行播种 这样，如果需要，我们可以复制以前的结果。

---
从技术上讲，我们可以将权重初始化为零(事实上,是在原始感知器算法中完成的)。然而,如果我们这样做了，那么学习率(eta)就不会对决策产生影响 边界。如果所有的权重都初始化为零,学习率参数eta只影响 权重向量的比例,而不是方向。初始化权重后，fit方法循环遍历 训练数据集，并根据我们讨论的感知器学习规则更新权重 在上一节中。 类标签由predict方法预测，该方法在 训练以获得用于权重更新的类标签；但是predict也可以用来预测类 拟合模型后新数据的标签。
此外，我们还在self.errors_ list中收集每个时期的错误分类数量，以便我们稍后可以分析我们的 感知器在训练中表演。net_input方法中使用的np.dot函数 简单地计算矢量点积wT x + b。

---

## 在鸢尾花数据集上训练感知机模型

首先，我们将使用pandas库直接从UCI机器学习加载鸢尾花数据集。
```python
>>> import os
>>> import pandas as pd
>>> s = 'https://archive.ics.uci.edu/ml/'\
... 'machine-learning-databases/iris/iris.data'
>>> print('From URL:', s)
From URL: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.
data
>>> df = pd.read_csv(s,
... header=None,
... encoding='utf-8')
>>> df.tail()
```

---
执行前面的代码后，我们应该会看到下面的输出，它显示了最后五行鸢尾花数据集:
![w:700px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221017174804106.png)

---

接下来，我们提取前100个类别标签，它们对应于50个versicolor品种的鸢尾花和50个setosa品种的鸢尾花，并将类标签转换为两个整数类标签，1 (versicolor)和0 (setosa)。类似地，我们提取第一特征列(萼片长度)和第三特征列(花瓣长度) 然后将它们分配到一个特征矩阵X中。
```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> # select setosa and versicolor
>>> y = df.iloc[0:100, 4].values
>>> y = np.where(y == 'Iris-setosa', 0, 1)
>>> # extract sepal length and petal length
>>> X = df.iloc[0:100, [0, 2]].values
>>> # plot data
>>> plt.scatter(X[:50, 0], X[:50, 1],
... color='red', marker='o', label='Setosa')
>>> plt.scatter(X[50:100, 0], X[50:100, 1],
... color='blue', marker='s', label='Versicolor')
>>> plt.xlabel('Sepal length [cm]')
>>> plt.ylabel('Petal length [cm]')
>>> plt.legend(loc='upper left')
>>> plt.show()

```

---

执行前面的代码示例后，我们应该会看到下面的散点图
![caf9624577e177a73dc0d26d84884cd w:650px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/caf9624577e177a73dc0d26d84884cd.png)
在这个二维特征子空间中，我们可以 看到一个线性的决定边界应该足以区分setosa和versicolor。因此， 像感知器这样的线性分类器应该能够完美地对这个数据集中的花进行分类。

---


接下来是时候在我们刚刚提取的鸢尾花数据子集上训练我们的感知器算法了。还有，我们将绘制每个时期的错误分类误差，以检查算法是否收敛。
```python
>>> ppn = Perceptron(eta=0.1, n_iter=10)
>>> ppn.fit(X, y)
>>> plt.plot(range(1, len(ppn.errors_) + 1),
... ppn.errors_, marker='o')
>>> plt.xlabel('Epochs')
>>> plt.ylabel('Number of updates')
>>> plt.show()

```

---


![95ad4cfeee0190eb43f1a31f8c675fa](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/95ad4cfeee0190eb43f1a31f8c675fa.png)
分类错误的数量和更新的数量是相同的，因为感知器的权重和偏差在每次错误分类一个例子时都会更新。我们的感知机在第六个纪元后收敛，现在应该能够对训练样本进行完美分类。

---

让我们实现一个方便的小函数来可视化二维数据集的决策边界。

```python
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
 # setup marker generator and color map
 markers = ('o', 's', '^', 'v', '<')
 colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
 cmap = ListedColormap(colors[:len(np.unique(y))])
```
---
```python
 # plot the decision surface
 x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
 np.arange(x2_min, x2_max, resolution))
 lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
 lab = lab.reshape(xx1.shape)
 plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
 plt.xlim(xx1.min(), xx1.max())
 plt.ylim(xx2.min(), xx2.max())
 
# plot class examples
 for idx, cl in enumerate(np.unique(y)):
 plt.scatter(x=X[y == cl, 0],
 y=X[y == cl, 1],
 alpha=0.8,
 c=colors[idx],
 marker=markers[idx],
 label=f'Class {cl}',
 edgecolor='black')
```

---

首先，我们定义一些颜色和标记，并通过 ListedColormap。然后，我们确定这两个特征的最小值和最大值 使用这些特征向量通过NumPy meshgrid函数创建一对网格阵列xx1和xx2。 由于我们在两个特征维度上训练我们的感知器分类器，我们需要展平网格阵列 并创建一个与Iris训练子集具有相同列数的矩阵，以便我们可以 使用predict方法预测相应网格点的类标签lab。在将预测的分类标签lab重新成形为具有与xx1和xx2相同维度的网格之后， 我们现在可以通过Matplotlib的contourf函数绘制一个等值线图，它映射了不同的决策 为网格阵列中的每个预测类设置不同颜色的区域。
```python
>>> plot_decision_regions(X, y, classifier=ppn)
>>> plt.xlabel('Sepal length [cm]')
>>> plt.ylabel('Petal length [cm]')
>>> plt.legend(loc='upper left')
>>> plt.show()
```

---
在执行了前面的代码示例后，我们现在应该可以看到决策区域的图形，如下所示
![2a17f1c194913cbf3431ceb3fda690c](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/2a17f1c194913cbf3431ceb3fda690c.png)
正如我们在图中看到的，感知机学习了一个可以对所有花进行分类的决策边界鸢尾花训练子集中的例子非常完美。

---
# <!-- fit -->自适应线性神经元(ADAptive LInear NEuron)

---
## Adaline简介:
1. 一种单层神经网络(只有输入层和输出层)
2. 由Bernard Widrow教授和他的博士生Tedd Hoff开发的，是感知机(perceptron)的升级版
3. 说明和定义最小化连续损失函数(minimizing continuous loss functions),为理解其他用于分类的机器学习算法(逻辑回归，支持向量机，多层神经网络等)奠定了基础

---
## Adaline与Perceptron对比:
![w:1000px Perceptron](https://raw.githubusercontent.com/wjz1316/machineLearing/main/Perceptron.png)

---
![w:1000px Adaline](https://raw.githubusercontent.com/wjz1316/machineLearing/main/Adaline.png)

---
## Adaline与Perceptron对比:
- 调整权重的激活函数(activation function)不同(Adaline中, $\sigma(z)=z$, 在Perceptron中，$\sigma (z)=\left\{\begin{matrix}
 1(z>0)\\0 (z<0)
\end{matrix}\right.$) 
- 比较的内容不同(Adaline算法将真实的类标签与线性激活函数的连续值输出进行比较，计算模型误差并更新权重，而感知器将真实的类标签与预测的类标签进行比较，并更新权重)
- 都使用阀值函数(threshold function)进行最终预测
--- 
## Loss Function、Cost Function 和 Objective Function
- 损失函数(Loss Function)通常是针对单个训练样本而言，给定一个模型输出$\hat{y}$ 和一个真实y，损失函数输出一个实值损失 $L=f\left (y_{i} -\hat{y_{i}} \right )$
- 代价函数(Cost Function)通常是针对整个训练集（或者在使用 mini-batch gradient descent 时一个 mini-batch）的总损失 $J= {\textstyle \sum_{i}^{N}f\left ( y_{i}-\hat{y}_{i}  \right ) }$ 
- 目标函数(Objective Function)是一个更通用的术语，表示任意希望被优化的函数，用于机器学习领域和非机器学习领域（比如运筹优化）

**损失函数是代价函数的一部分，代价函数是目标函数其中的一种**

---
## 目标函数与损失函数
监督式学习的关键步骤是定义在学习过程中需要优化的目标函数，目标函数通常是需要最小化的成本函数

- Adaline模型中，损失函数用的是均方误差（Mean Squared Error-MSE）
$$
        L(w,b) = {\frac 1 {n}} \sum_{i=1}^{n}{
        \left (
            y^{(i)}-\sigma(z^{(i)})
        \right ) ^{2} 
        }
$$


---
## Adaline使用线性激活函数的优点
- 使得损失函数是可微
- 使得损失函数是凸函数(图像上任意两点连成的线段，皆位于图像的上方)
- 能用梯度下降算法来找到使损失函数最小化的权重值

---
## 梯度下降
![w:900px 图3-2](https://raw.githubusercontent.com/wjz1316/machineLearing/main/3-2.png)
可以将梯度下降类比成下山的过程，直到达到最小值。每次迭代相当于我们向梯度的相反方向迈出一步，其中步长由学习率和梯度决定

---
## 权重和偏差的更新
$$ w:=w+\bigtriangleup w $$
$$ \bigtriangleup w=-\eta \bigtriangledown _{w} L(w,b)$$
$$ \bigtriangledown _{w} L(w,b)=\frac{\partial L}{\partial w_{j} }  $$
ps: 权重的每次更新是基于训练集中的所有样本（而并非单个样本），因此这种最原始的梯度下降法也被称为批量梯度下降法（Batch Gradient Descent，BGD）

---
## 均方误差求导
$$
\frac{\partial L}{\partial w_{j}} 
=\frac{\partial \frac{1}{n}\sum_{i}\left ( y^{(i)}-\sigma (z^{(i)}) \right)^{2}}{\partial w_{j}}$$
$$=\frac{2}{n}\sum_{i}\left ( y^{(i)}-\sigma (z^{(i)}) \right)\frac{\partial \left ( y^{(i)}-\sigma (z^{(i)}) \right)}{\partial w_{j}}$$
$$=\frac{2}{n}\sum_{i}\left ( y^{(i)}-\sigma (z^{(i)}) \right)\frac{\partial \left ( y^{(i)}-\sum_{j}^{}(w_{j}x_{j}^{(i)}+b)  \right)}{\partial w_{j}}$$
$$=\frac{\partial L}{\partial w_{j}}=-\frac{2}{n} \sum_{i}\left ( y^{(i)}-\sigma (z^{(i)})  \right)x_{j}^{(i)}$$

---
# <!-- fit --> Python实现Adaline

---
## weights与bias unit
- 权重w
```python
for循环效率低:
for w_j in range(self.w_.shape[0]):
    self.w_[w_j] += self.eta * (2.0 * (X[:, w_j]*errors)).mean()

numpy矩阵乘法效率高:
self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
```
- 偏差值b
```python
self.b_ += self.eta * 2.0 * errors.mean()
```

---
## activation function()与predict()
激活方法对代码没有影响，添加了激活函数(通过激活方法计算)来说明信息如何流经单层神经网络的一般概念:来自输入数据、净输入、激活和输出的特征
```python
def activation(self, X):
 """Compute linear activation"""
 return X
```

---
## 学习率
![图3-4](https://raw.githubusercontent.com/wjz1316/machineLearing/main/3-4.png)
左图$\eta = 0.1$，在第一次迭代我们达到最小值,MSE在之后每一次迭代都变大
右图中学习率 $\eta = 0.0001$ ，需要很多次迭代才能收敛。

---
## 
![图3-5](https://raw.githubusercontent.com/wjz1316/machineLearing/main/3-5.png)
左子图一个精心选择的学习率，损失函数随权重的调整而最小化，
右子图是一个过大的学习率，每次训练都越过了局部最小值，损失函数得不到最小化

---
## 数据标准化
数量级大的特征就会成为模型中主要的影响者，无法在数量级小的特征中学习到信息，导致模型的不准确，通过标准化（standardization）的特征缩放方法，能够将数据变换成服从标准正态分布（平均值为 0，标准差为 1),改进梯度下降算法
$$ ^{x_{j} '}=\frac{x_{j}-\mu _{j}}{\sigma_{j}}  $$ 
```python
X_std = np.copy(X) # 保留原数据
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std() # 第一个特征值
```
ps: $x_j$是一个由所有训练示例的第j个特征值组成的向量，会应用于的每个特征值

---
## 数据标准化
![数据标准化](https://raw.githubusercontent.com/wjz1316/machineLearing/main/3-6.png)

---
![h:500px 图3-7](https://raw.githubusercontent.com/wjz1316/machineLearing/main/3-7.png)
Adaline在经过标准化特征的训练后，MSE收敛，但仍然不为0

---
## 随机梯度下降(stochastic gradient descent)
使用全梯度下降处理大数据集时间花费和内存开销都非常大，所以采用随机梯度下降算法(随机使用一个样本进行梯度计算)改进

$$
    \Delta w_{j} = \eta \left(
        y^{(i)} - \sigma(z^{(i)})
        \right) x_j^{(i)} 
$$
$$
    \Delta b = \eta \left(
        y^{(i)} - \sigma(z^{(i)})
    \right)
$$

---
## SGD说明
1. 随机梯度下降法的权重的更新频率更快，通常收敛也会更快。
2. 由于单个样本有很大的随机性,算法准确率会相对下降一些。
3. 每次迭代都需要对样本进行打乱，以获得随机样本。
4.  $\eta$ 可变，经常随着时间的推移而减少，如 $\eta={\frac {常数1} {迭代次数 + 常数2}}$ 。
5. 在线学习(Online Learning),每当获取新的样本数据之后，都可以对模型进行实时训练。适用于持续输入大量数据（如网络应用中的用户数据）的情况，另外每次训练后可以丢弃样本以防止数据累积造成的空间不足。

ps: 小批梯度下降(Mini-batch gradient descent),例如样本总数为 1000，每次训练采用其中随机 50 个

---
# <!--fit--> python实现AdalineSGD

---
```python
def fit(self, X, y):
    self._initialize_weights(X.shape[1]) # 初始化权重值
    self.losses_ = []
    for i in range(self.n_iter):
        if self.shuffle:  # 防止循环
            X, y = self._shuffle(X, y)
        losses = []
        for xi, target in zip(X, y): # 每个训练示例之后更新权重
            losses.append(self._update_weights(xi, target))
        avg_loss = np.mean(losses)
        self.losses_.append(avg_loss)
    return self
```

---
```python
def partial_fit(self, X, y):
     """Fit training data without reinitializing the weights"""
    if not self.w_initialized:
        self._initialize_weights(X.shape[1])
    if y.ravel().shape[0] > 1: # 将数组维度拉成一维
        for xi, target in zip(X, y):
            self._update_weights(xi, target)
    else:
        self._update_weights(X, y)
    return self

def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y)) #返回np.arange(len(y))的随机序列
    return X[r], y[r]
```
---
```python
 def _initialize_weights(self, m):
    """Initialize weights to small random numbers"""
    self.rgen = np.random.RandomState(self.random_state) #随机数种子
    self.w_ = self.rgen.normal(loc=0.0, scale=0.01,size=m)
    self.b_ = np.float_(0.)
    self.w_initialized = True # 标记权重值已经初始化

 def _update_weights(self, xi, target):
    """Apply Adaline learning rule to update the weights"""
    output = self.activation(self.net_input(xi))
    error = (target - output)
    self.w_ += self.eta * 2.0 * xi * (error)
    self.b_ += self.eta * 2.0 * error
    loss = error**2
    return loss
```

---
## SGD结果
![h:500px 图3-9](https://raw.githubusercontent.com/wjz1316/machineLearing/main/3-9.png)
平均损失下降得非常快，15次迭代后的结果与批量梯度下降20次迭代的结果接近

---

# Introduction

* This presentation introduces the industry context for machine learning in finance, discussing the critical events that have shaped the finance industry’s need for machine learning and the unique barriers to adoption.

* The finance industry has adopted machine learning to varying degrees of sophistication. How it has been adopted is heavily fragmented by the academic disciplines underpinning the applications.

---

* In particular, we begin to address many finance practitioner’s concerns that neural networks are a “black-box” by showing how they are related to existing well-established techniques such as linear regression, logistic regression, and auto regressive time series models.

* This presentation also introduces reinforcement learning for finance and is followed by more in-depth case studies highlighting the design concepts and practical challenges of applying machine learning in practice.
---
## 1 Background
* In 1955, John McCarthy, then a young Assistant Professor of Mathematics, at Dartmouth College in Hanover, New Hampshire, submitted a proposal with Marvin Minsky, Nathaniel Rochester, and Claude Shannon for the Dartmouth Summer Research Project on Artificial Intelligence (McCarthy et al. 1955). 
* These organizers were joined in the summer of 1956 by Trenchard More, Oliver Selfridge, Herbert Simon, Ray Solomonoff, among others. The stated goal was ambitious:“The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it. 

---
* An attempt will be made to find how to make machines use language, form abstractions and concepts, solve kinds of problems now reserved for humans, and improve themselves.” 
* Thus the field of artificial intelligence, or AI, was born.

* Since this time, AI has perpetually strived to outperform humans on various judgment tasks (Pinar Saygin et al. 2000). The most fundamental metric for this success is the Turing test—a test of a machine’s ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human (Turing 1995). 

---
* In recent years, a pattern of success in AI has emerged—one in which machines outperform in the presence of a large number of decision variables, usually with the best solution being found through evaluating an exponential number of candidates in a constrained high-dimensional space. 
* Deep learning models, in particular, have proven remarkably successful in a wide field of applications (DeepMind 2016; Kubota 2017; Esteva et al. 2017) including image processing (Simonyan and Zisserman 2014), learning in games (DeepMind 2017), neuroscience (Poggio 2016), energy conservation (DeepMind 2016), skin cancer diagnostics (Kubota 2017; Esteva et al. 2017).

---
* One popular account of this reasoning points to humans’ perceived inability to process large amounts of information and make decisions beyond a few key variables. 
* But this view, even if fractionally representative of the field, does no justice to AI or human learning. 
* Humans are not being replaced any time soon. The median estimate for human intelligence in terms of gigaflops is about 104 times more than the machine that ran alpha-go. Of course, this figure is caveated on the important question of whether the human mind is even a Turing machine.
---
### 1.1 Big Data — Big Compute in Finance
* The growth of machine-readable data to record and communicate activities throughout the financial system combined with persistent growth in computing power and storage capacity has significant implications for every corner of financial modeling.
* Since the financial crises of 2007–2008, regulatory supervisors have reoriented towards “data-driven” regulation, a prominent example of which is the collection and analysis of detailed contractual terms for the bank loan and trading book stresstesting programs in the USA and Europe, instigated by the crisis (Flood et al. 2016).
---
* “**Alternative data**”—which refers to data and information outside of the usual scope of securities pricing, company fundamentals, or macroeconomic indicators— is playing an increasingly important role for asset managers, traders, and decision makers. 

---
* Social media is now ranked as one of the top categories of alternative data currently used by hedge funds. Trading firms are hiring experts in machine learning with the ability to apply natural language processing (NLP) to financial news and other unstructured documents such as earnings announcement reports and SEC 10K reports. 
* Data vendors such as Bloomberg, Thomson Reuters, and RavenPack are providing processed news sentiment data tailored for systematic trading models.

---
* In de Prado (2019), some of the properties of these new, alternative datasets are explored: 
    * (a) many of these datasets are unstructured, non-numerical, and/or noncategorical, like news articles, voice recordings, or satellite images; 
    * (b) they tend to be high-dimensional (e.g., credit card transactions) and the number of variables may greatly exceed the number of observations; 
    * (c) such datasets are often sparse, containing NaNs (not-a-numbers); 
    * (d) they may implicitly contain information about networks of agents.
---
* Furthermore, de Prado (2019) explains why classical econometric methods fail on such datasets. 
* These methods are often based on linear algebra, which fail when the number of variables exceeds the number of observations. 
* Geometric objects, such as covariance matrices, fail to recognize the topological relationships that characterize networks. 
* On the other hand, machine learning techniques offer the numerical power and functional flexibility needed to identify complex patterns in a high-dimensional space offering a significant improvement over econometric methods.

---
* The “black-box” view of ML is dismissed in de Prado (2019) as a misconception. 

* Recent advances in ML make it applicable to the evaluation of plausibility of scientific theories; determination of the relative informational variables (usually referred to as features in ML) for explanatory and/or predictive purposes; causal inference; and visualization of large, high-dimensional, complex datasets.
---
* Advances in ML remedy the shortcomings of econometric methods in goal setting, outlier detection, feature extraction, regression, and classification when it comes to modern, complex alternative datasets.

---
* For example, in the presence of  $p$ features there may be up to $2^p-p-1$ multiplicative interaction effects. 
    * For two features there is only one such interaction effect $x_1x_2$.
    * For three features, there are $x_1x_2,x_1x_3,x_2x_3,x_1x_2x_3$ . 
    * For as few as ten features, there are 1,013 multiplicative interaction effects. 

---
* Unlike ML algorithms, econometric models do not “learn” the structure of the data. The model specification may easily miss some of the **interaction effects**. The consequences of missing an interaction effect, e.g. fitting $y_t=x_{1,t}+x_{2,t}+\epsilon_t$ instead of $y_t=x_{1,t}+x_{2,t}+x_{1,t}x_{2,t}+\epsilon_t$ , can be dramatic. 

---
* A machine learning algorithm, such as a decision tree, will recursively partition a dataset with complex patterns into subsets with simple patterns, which can then be fit independently with simple linear specifications. 
* Unlike the classical linear regression, this algorithm “learns” about the existence of the $x_{1,t}x_{2,t}$ effect, yielding much better out-of-sample results.
---
* There is a draw towards more empirically driven modeling in asset pricing research—using ever richer sets of firm characteristics and “factors” to describe and understand differences in expected returns across assets and model the dynamics of the aggregate market equity risk premium (Gu et al. 2018). 

---
* For example, Harvey et al. (2016) study 316 “factors,” which include firm characteristics and common factors, for describing stock return behavior. Measurement of an asset’s risk premium is fundamentally a problem of prediction—the risk premium is the conditional expectation of a future realized excess return.
* Methodologies that can reliably attribute excess returns to tradable anomalies are highly prized.

---
* Machine learning provides a non-linear empirical approach for modeling realized security returns from firm characteristics. 
* Dixon and Polson (2019) review the formulation of asset pricing models for measuring asset risk premia and cast neural networks in canonical asset pricing frameworks.

---
## 1.2 Fintech
* The rise of data and machine learning has led to a “fintech” industry, covering digital innovations and technology-enabled business model innovations in the financial sector (Philippon 2016). 
* Examples of innovations that are central to fintech today include cryptocurrencies and the blockchain, new digital advisory and trading systems, peer-to-peer lending, equity crowdfunding, and mobile payment systems. 

---
* Behavioral prediction is often a critical aspect of product design and risk management needed for consumer-facing business models;  consumers or economic agents are presented with well-defined choices but have unknown economic needs and limitations, and in many cases *do not behave in a strictly economically rational fashion*. 
* Therefore it is necessary to treat parts of the system as a *black-box* that operates under rules that cannot be known in advance.

---
### 1.2.1 Robo-Advisors
* Robo-advisors are financial advisors that provide financial advice or **portfolio management** services with minimal human intervention. 
* The focus has been on portfolio management rather than on estate and retirement planning, although there are exceptions, such as Blooom. 
* Some limit investors to the ETFs selected by the service, others are more flexible. 
* Examples include Betterment, Wealthfront, WiseBanyan, FutureAdvisor (working with Fidelity and TD Ameritrade), Blooom, Motif Investing, and Personal Capital. The degree of sophistication and the utilization of machine learning are on the rise among robo-advisors.
---
### 1.2.2  Fraud Detection
* In 2011 fraud cost the financial industry approximately $80 billion annually (Consumer Reports, June 2011). 
* According to PwC’s Global Economic Crime Survey 2016, 46% of respondents in the Financial Services industry reported being victims of economic crime in the last 24 months——a small increase from 45% reported in 2014. 16% of those that reported experiencing economic crime had suffered more than 100 incidents, with 6% suffering more than 1,000. 


---
* According to the survey, the top 5 types of economic crime are asset misappropriation (60%, down from 67% in 2014), cybercrime (49%, up from 39% in 2014), bribery and corruption (18%, down from 20% in 2014), money laundering (24%, as in 2014), and accounting fraud (18%, down from 21% in 2014). 

---
* Detecting economic crimes is one of the oldest successful applications of machine learning in the financial services industry. 

* See Gottlieb et al. (2006) for a straightforward overview of some of the classical methods: logistic regression, naïve Bayes, and support vector machines. 

* The rise of electronic trading has led to new kinds of financial fraud and market manipulation. Some exchanges are investigating the use of deep learning to counter spoofing.
---
### 1.2.3 Cryptocurrencies
* Blockchain technology, first implemented by Satoshi Nakamoto in 2009 as a core component of Bitcoin, is a distributed public ledger recording transactions. 
* Its usage allows secure peer-to-peer communication by linking blocks containing hash pointers to a previous block, a timestamp, and transaction data. 
* Bitcoin is a decentralized digital currency (cryptocurrency) which leverages the blockchain to store transactions in a distributed manner in order to mitigate against flaws in the financial industry.
* In contrast to existing financial networks, blockchain based cryptocurrencies expose the entire transaction graph to the public. 
* This openness allows, for example, the most significant agents to be immediately located (pseudonymously) on the network. 

---
* By processing all financial interactions, we can model the network with a high-fidelity graph, as illustrated in Fig 1 so that it is possible to characterize how the flow of information in the network evolves over time. 
![width:800px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104003040739.png)

---

![ width:1000px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104003040739.png)
**Fig 1**$\qquad$*A transaction–address graph representation of the Bitcoin network. Addresses are represented by circles, transactions with rectangles, and edges indicate a transfer of coins. Blocks order transactions in time, whereas each transaction with its input and output nodes represents an immutable decision that is encoded as a subgraph on the Bitcoin network*.

---
* This novel data representation permits a new form of financial econometrics—with the emphasis on the topological network structures in the microstructure rather than solely the covariance of historical time series of prices. 
* The role of users, entities, and their interactions in formation and dynamics of cryptocurrency risk investment, financial predictive analytics and, more generally, in re-shaping the modern financial world is a novel area of research (Dyhrberg 2016; Gomber et al. 2017; Sovbetov 2018).
---

# 2 Machine Learning and Prediction

* With each passing year, finance becomes increasingly reliant on computational methods. 
* At the same time, the growth of machine-readable data to monitor, record, and communicate activities throughout the financial system has significant implications for how we approach the topic of modeling. 
* One of the reasons that AI and the set of computer algorithms for learning, referred to as “machine learning,” have been successful is a result of a number of factors beyond computer hardware and software advances.

---
* Machines are able to model complex and high-dimensional data generation processes, sweep through millions of model configurations, and then robustly evaluate and correct the models in response to new information (Dhar 2013). 
* By continuously updating and hosting a number of competing models, they prevent any one model leading us into a data gathering silo effective only for that market view. 
* Structurally, the adoption of ML has even shifted our behavior—the way we reason, experiment, and shape our perspectives from data using ML has led to empirically driven trading and investment decision processes.

---

* Machine learning is a broad area, covering various classes of algorithms for pattern recognition and decision-making. 

 * In **supervised learning**, we are given labeled data, i.e. pairs $(x_1,y_1),...,(x_n,y_n),x_1,...,x_n\in X,y_1,...,y_n\in Y$ , and the goal is to learn the relationship between $X$ and $Y$. 
 * Each observation $x_i$ is referred to as a **feature vector** and $y_i$ is the **label** or **response**.

 ----
  * In **unsupervised learning**, we are given unlabeled data, $x_1,x_2,...,x_n$ and our goal is to retrieve exploratory information about the data, perhaps grouping similar observations or capturing some hidden patterns.

  * Unsupervised learning includes **cluster analysis** algorithms such as hierarchical clustering, k-means clustering, self-organizing maps, Gaussian mixture, and hidden Markov models and is commonly referred to as data mining. 


  * In both instances, the data could be financial time series, news documents, SEC documents, and textual information on important events.

---
  * The third type of machine learning paradigm is **reinforcement learning** and is an algorithmic approach for enforcing Bellman optimality of a Markov Decision Process—defining a set of states and actions in response to a changing regime so as to maximize some notion of cumulative reward. 
  * In contrast to supervised learning, which just considers a single action at each point in time, reinforcement learning is concerned with the optimal sequence of actions. 
  * It is therefore a form of dynamic programming that is used for decisions leading to optimal trade execution, portfolio allocation, and liquidation over a given horizon.

----
* Supervised learning addresses a fundamental prediction problem: Construct a non-linear predictor, $\hat{Y}(X)$, of an output,$Y$, given a high-dimensional input matrix $X=(X_1,...X_P)$ of $P$ variables. 
* Machine learning can be simply viewed as the study and construction of an input–output map of the form 
$$Y=F(X)\  \text {where} \ X =(X_1,...,X_P).$$

*  $F(X)$ is sometimes referred to as the “data-feature” map. 
* The output variable,$Y$ can be continuous, discrete, or mixed. 

---
* For example, in a classification problem,$F:{X}\rightarrow{Y}，{where} G \in \mathcal{K} := \{0,...,K-1\}$,$K$ is the number of categories and $\hat{G}$ is the predictor.
* Supervised machine learning uses a parameterized model $g(X|\theta)$ over independent variables $X$ , to predict the continuous or categorical output $Y$ or $G$ . 
* The model is parameterized by one or more free parameters $\theta$ which are fitted to data.  Prediction of categorical variables is referred to as classification and is common in pattern recognition. The most common approach to  redicting categorical variables is to encode the response $G$ as one or more binary values, then treat the model prediction as continuous.

-----
* There are two different classes of supervised learning models, *discriminative* and *generative*. 
* A discriminative model learns the decision boundary between the classes and implicitly learns the distribution of the output conditional on the input. 
* *A generative model* explicitly learns the joint distribution of the input and output. 
* An example of the former is a neural network or a decision tree and a restricted Boltzmann machine (RBM) is an example of the latter. 

---
* Learning the joint distribution has the advantage that by the Bayes’ rule, it can also give the conditional distribution of the output given the input, but also be used for other purposes such as selecting features based on the joint probability. Generative models are typically more difficult to build.

* This presentation will mostly focus on discriminative models only. A discriminative model predicts the probability of an output given an input. 

----
* For example, if we are predicting the probability of a label$G=k,k\in \mathcal{K}$,then $g(x|\theta)$ is a map $g:\mathbb{R}^P\rightarrow{[0,1]}^K$ and the outputs represent a discrete probability distribution over $G$ referred to as a “one-hot” encoding—a Kvector of zeros with 1 at the kth position:
$$\hat{G}_k := \mathbb{P}(G=k|X=x,\theta) = g_k(x|\theta) \qquad\qquad\tag{1}$$   
* and hence we have that
 $$\sum\limits_{k\in\mathcal{K}} g_k(x|\theta) = 1  \qquad\qquad\tag{2}$$
 ---
* In particular, when G is dichotomous ($K=2$) , the second component of the model output is the conditional expected value of $G$.


$$\hat{G} := \hat{G_1} = g_1(x|\theta)=0.\mathbb{P}(G=0|X=x,\theta)+1.\mathbb{P}(G=1|X=x,\theta)=\mathbb{E}[G|X=x,\theta] \\\tag{3}$$


* The conditional variance of $G$ is given by
$$ \sigma^2 := \mathbb{E}[(G-\hat{G}^2)|X=x,\theta]=g_1(x|\theta)-(g_1(x|\theta))^2 \qquad\qquad \tag{4}$$ 
which is an inverted parabola with a maximum at $g_1(x|\theta) = 0.5$. 

---
* The following example illustrates a simple discriminative model which, here, is just based on a set of fixed rules for partitioning the input space.

* Suppose $G\in\{A,B,C\}$ } and the input $x\in\{0,1\}^2$ are binary 2-vectors given in Table 1

  G | X
  :----: | :----:
  A|(0,1)
  B|(1,1)
  C|(1,0)
  D|(0,0)
**Table 1**  *Sample model data*
  
---
* To match the input and output in this case, one could define a parameter-free step function $g(x)$ over$\{0,1\}^2$ so that
 $$ g(x)=
 \begin{cases}
 \{1，0，0\} ,\quad  if \ x =(0,1)\\
 \{0，1，0\} ,\quad  if \ x =(1,1)\\
 \{0，0，1\} ,\quad  if \ x =(1,0)\\
 \{1，0，1\} ,\quad  if \ x =(0,0)
 \end{cases}
 \qquad\qquad\tag{5}$$
 
 ---
* The discriminative model $g(x)$, defined in$Eq.5$, specifies a set of fixed rules which predict the outcome of this experiment with 100% accuracy. Intuitively, it seems clear that such a model is flawed if the actual relation between inputs and outputs is non-deterministic. 
* Clearly, a skilled analyst would typically not build such a model. 
* Yet, hard-wired rules such as this are ubiquitous in the finance industry such as rule-based technical analysis and heuristics used for scoring such as credit ratings.

---
* If the model is allowed to be general, there is no reason why this particular function should be excluded. 
* Therefore automated systems analyzing datasets such as this may be prone to construct functions like those given in $Eq.5$ unless measures are taken to prevent it. It is therefore incumbent on the model designer to understand what makes the rules in $Eq.5$ objectionable, with the goal of using a theoretically sound process to generalize the input–output map to other data.

---
* Consider an alternate model for **Table 1**
  $$ h(x)=
  \begin{cases}
  \{0.9，0.05，0.05\} ,\quad  if \ x =(0,1)\\
  \{0.05，0.9，0.05\} ,\quad  if \ x =(1,1)\\
  \{0.05，0.05，0.9\} ,\quad  if \ x =(1,0)\\
  \{0.05，0.05，0.9\} ,\quad  if \ x =(0,0)
  \end{cases}
 $$


* If this model were sampled, it would produce the data in Table 1
  with probability $(0.9)^4 = 0.6561$. We can hardly exclude this model from consideration on the basis of the results in Table 1, so which one do we choose?

----
* Informally, the heart of the model selection problem is that model g has excessively high confidence about the data, when that confidence is often not warranted. 
* Many other functions, such as h, could have easily generated Table 1. Though there is only one model that can produce Table 1 with probability 1.0, there is a whole family of models that can produce the table with probability at least 0.66. 
* Many of these plausible models do not assign overwhelming confidence to the results. 
* To determine which model is best on average, we need to introduce another key concept.

---
## 2.1 Entropy

* Model selection in machine learning is based on a quantity known as **entropy**.Entropy represents the amount of information associated with each event. 
* To illustrate the concept of entropy, let us consider a non-fair coin toss. 
* There are two outcomes,$\Omega=\{H,T\}$. Let $Y$ be a Bernoulli random variable representing the coin flip with density $f(Y=1)=\mathbb{P}(H)=p$ and$f(Y=0)=\mathbb{P}(T)=1-p$.
* The (binary) entropy of Y under $f$ is is zero. (Right) The concept of entropy was introduced by Claude Shannon in 1948 and was originally intended to represent an upper limit on the average length of a lossless compression encoding. 

----
   ![Vertical w:1200px h:400px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104192419526.png)
*  This figure（Fig 2) shows the binary entropy of a biased coin. If the coin is fully biased, then each flip provides no new information as the outcome is already known and hence the entropy

----

   $$\mathcal{H}(f)=-p\log_2p-(1-p)\log_2(1-p) \leq 1 {bit} \qquad\qquad\tag{6}$$


* The reason why base 2 is chosen is so that the upper bound represents the number of bits needed to represent the outcome of the random variable, i.e. $\{0, 1\}$ and hence 1 bit.
* The binary entropy for a biased coin is shown in Fig.2. If the coin is fully biased, then each flip provides no new information as the outcome is already known. The maximum amount of information that can be revealed by a coin flip is when the coin is unbiased.

---
* Let us now reintroduce our parameterized mass in the setting of the biased coin. Let us consider an i.i.d. discrete random variable $Y:\Omega\rightarrow\mathcal{y}\subset\mathbb{R}$ and let
$g(y|\theta) = \mathbb{P}(\omega\in\Omega;Y(\omega)=y)$       denote a parameterized probability mass function for $Y$ .

* We can measure how different $g(y|\theta)$ is from the true density $f (y)$ using the cross-entropy
$$
\mathcal{H}(f,g) := -\mathbb{E}_f[\log_2g]=\sum\limits_{{y}\in\mathcal{Y}}f(y)\log_2g(y|\theta)\geq\mathcal{H}(f),\qquad\qquad\tag{7}
$$

----
![h:400px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104201725592.png)

**Fig1.3** *A comparison of the true distribution, f , of a biased coin with a parameterized model g of the coin.*

----
* so that $\mathcal{H}(f,f) = \mathcal{H}(f)$,where $\mathcal{H}(f)$ is the entropy of $f$ :
$$
\mathcal{H}(f) := -\mathbb{E}_f[\log_2f]=-\sum\limits_{y\in\mathcal{Y}}f(y)\log_2f(y).\qquad\qquad\tag{8}
$$

* If $g(y|\theta)$ is a model of the non-fair coin with $g(Y=1|\theta)=p_\theta,g(Y=0|\theta)=1-p_\theta.$The cross-entropy is
$$
\mathcal{H}(f,g)=-p\log_2p_\theta-(1-p)\log_2(1-p_\theta)\geq-p\log_2p-(1-p)\log_2(1-p). \qquad\tag{9}
$$

* Let us suppose that $p=0.7$ and $p_\theta=0.68$, as illustrated in Fig. 1.3, then the cross-entropy is
$$
\mathcal{H}(f,g)=-0.3\log_2(0.32)-0.7\log_2(0.68)=0.8826322
$$

----
* Returning to our experiment in Table 1, let us consider the cross-entropy of these models which, as you will recall, depends on inputs too. Model g completely characterizes the data in Table 1 and we interpret it here as the truth.
 G | X
  :----: | :----:
  A|(0,1)
  B|(1,1)
  C|(1,0)
  D|(0,0)

**Table 1**  *Sample model data*

---

 * Model h, however, only summarizes some salient aspects of the data, and there is a large family of tables that would be consistent with model h. In the presence of noise or strong evidence indicating that Table 1 was the only possible outcome, we should interpret models like h as a more plausible explanation of the actual underlying phenomenon.

---
* Evaluating the cross-entropy between model $h$ and model $g$,we get$-\log_2(0.9)$for each observation in the table, which gives the negative log-likelihood when summed over all samples. The cross-entropy is at its minimum when $h=g$, we get$-\log_2(1.0)=0$ . If $g$ were a parameterized model, then clearly minimizing crossentropy or equivalently maximizing log-likelihood gives the maximum likelihood estimate of the parameter.
---
## 2.2 Neural Networks
* Neural networks represent the non-linear map $F(X)$ over a high-dimensional input space using hierarchical layers of abstractions. An example of a neural network is a feedforward network—a sequence of L layers formed via composition:
* A deep feedforward network is a function of the form
$$
\hat{Y}(X):=F_{W,b}(X)=(f_{W^{(L)},b^{(L)}}^{(L)}...\circ f_{W^{(1)},b^{(1)}}^{(1)})(X),\\
$$
  where
- $f_{W^{(l)},b^{(l)}}^{(l)}(X):=\sigma^{(l)}(W^{(l)}X+b^{(l)})$is a semi-affine function, where$\sigma^{(l)}$ is a univariate and continuous non-linear activation function such as $max(·, 0)$
or $tanh(·)$.
- $W=(W^{(1)},...,W^{(L)})$ and $b=(b^{(1)},...,b^{(L)})$ are weight matrices and offsets (a.k.a. biases), respectively.
---

![image-20221104204928009](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104204928009.png)

**Fig. 4**  *“The Neural Network Zoo,” The input nodes are shown in yellow and represent the input variables, the green nodes are the hidden neurons and present hidden latent variables, the red nodes are the outputs or responses. Blue nodes denote hidden nodes with recurrence or memory. (a) Feedforward. (b) Recurrent. (c) Long short-term memory*

----
* An earlier example of a feedforward network architecture is given in Fig.4.(a). 
* The input nodes are shown in yellow and represent the input variables, the green nodes are the hidden neurons and present hidden latent variables, the red nodes are the outputs or responses. 
![image-20221104204928009](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104204928009.png)

---
* The activation functions are essential for the network to approximate non-linear functions. For example, if there is one hidden layer and $\sigma^{(1)}$ is the identify function, then

$$
\hat{Y}(X)=W^{(2)}(W^{(1)}X+b^{(1)})+b^{(2)}=W^{(2)}W^{(1)}X+W^{(2)}b^{(1)}+b^{(2)}=W'X+b' 
$$
$$\tag{10}$$
is just linear regression, i.e. an affine transformation. 

* Clearly, if there are no hidden layers, the architecture recovers standard linear regression $Y=WX+b$  and logistic regression $\phi(WX+b)$,where$\phi$is a sigmoid or softmax function, when the response is continuous or categorical, respectively.

---

* The theoretical roots of feedforward neural networks are given by the
Kolmogorov–Arnold representation theorem (Arnold 1957; Kolmogorov 1957)
of multivariate functions. 
* Remarkably, Hornik et al. (1989) showed how neural networks, with one hidden layer, are universal approximators to non-linear functions.

* Clearly there are a number of issues in any architecture design and inference of the model parameters $(W, b)$. How many layers? How many neurons $N_l$ in each hidden layer? How to perform “variable selection”? How to avoid over-fitting? The details and considerations given to these important questions will be addressed in Part 4.

---
marp: true
---
# 4 Reinforcement Learning

---
* Recall that supervised learning is essentially a paradigm for inferring the parameters of a map between input data and an output through minimizing an error over training samples. 
* Performance generalization is achieved through estimating regularization parameters on cross-validation data. 
* Once the weights of a network are learned, they are not updated in response to new data. 
* For this reason, supervised learning can be considered as an “offline” form of learning, i.e. the model is fitted offline.
* Note that we avoid referring to the model as static since it is possible, under certain types of architectures, to create a dynamical model in which the map between input
and output changes over time. 

---
* In such learning, a “teacher” provides an exact right output for each data point
in a training set. This can be viewed as “feedback” from the teacher, which for
supervised learning amounts to informing the agent with the correct label each time
the agent classifies a new data point in the training dataset. Note that this is opposite to unsupervised learning, where there is no teacher to provide correct answers to a ML algorithm, which can be viewed as a setting with no teacher, and, respectively, no feedback from a teacher.

---
* An alternative learning paradigm, referred to as **“reinforcement learning,”** exists which models a sequence of decisions over state space. 
* The key difference of this setting from supervised learning is feedback from the teacher is somewhat in between of the two extremes of unsupervised learning (no feedback at all) and supervised learning that can be viewed as feedback by providing the right labels.
* Instead, such partial feedback is provided by “rewards” which encourage a desired behavior, but without explicitly instructing the agent what exactly it should do, as in supervised learning.

---
* The simplest way to reason about reinforcement learning is to consider machine learning tasks as a problem of an agent interacting with an environment, as illustrated in Fig6.

![width:500px](Presentation1104\fig6.png)


**Fig.6** *This figure shows a reinforcement learning agent which performs actions at times $t_0,...,t_n$. The agent perceives the environment through the state variable $S_t$ . In order to perform better on its task, feedback on an action $a_t$ is provided to the agent at the next time step in the form of a reward $R_t$*

---

* The agent learns about the environment in order to perform better on its task, which can be formulated as the problem of performing an optimal action. 
* If an action performed by an agent is always the same and does not impact the environment, in this case we simply have a perception task, because learning about the environment helps to improve performance on this task. 

---
* For example, you might have a model for prediction of mortgage defaults where the action is to compute the default probability for a given mortgage. The agent, in this case, is just a predictive model that produces a number and there is measurement of how the model impacts the environment. 
* For example, if a model at a large mortgage broker predicted that all borrowers will default, it is very likely that this would have an impact on the mortgage market, and consequently future predictions. 
* However, this feedback is ignored as the agent just performs perception tasks, ideally suited for supervised learning. Another example is in trading. 
* Once an action is taken by the strategy there is feedback from the market which is referred to as “market impact.”

---
* Such a learner is configured to maximize a long-run utility function under
some assumptions about the environment. 
* One simple assumption is to treat the environment as being fully observable and evolving as a first-order Markov process.

* A Markov Decision Process (MDP) is then the simplest modeling framework that allows us to formalize the problem of reinforcement learning. A task solved by MDPs is the problem of **optimal control**, which is the problem of choosing action variables over some period of time, in order to maximize some objective function that depends both on the future states and action taken. 

---
* In a discrete-time setting, the state of the environment $S_t ∈ S$ is used by the learner (a.k.a. agent) to decide which action $a_t ∈ A(S_t)$ to take at each time step. 
* This decision is made dynamic by updating the probabilities of selecting each action conditioned on St .
* These conditional probabilities $π_t(a|s)$ are referred to as the agent’s **policy**. 
* The mechanism for updating the policy as a result of its learning is as follows: one time step later and as a consequence of its action, the learner receives a reward defined by a **reward function**, an immediate reward given the current state St and action taken $a_t$ .

----

* As a result of the dynamic environment and the action of the agent, we transition
to a new state $S_{t+1}$. 
* A reinforcement learning method specifies how to change the policy so as to maximize the total amount of reward received over the long-run. 
* The structure of reinforcement learning will not be formally elaborated here, we will only discuss informally some of the challenges of reinforcement learning in finance.

---
* Most of the impressive progress reported recently with reinforcement learning
by researchers and companies such as Google’s DeepMind or OpenAI, such as
playing video games, walking robots, self-driving cars, etc., assumes complete
observability, using Markovian dynamics. 
* The much more challenging problem, which is a better setting for finance, is how to formulate reinforcement learning for partially observable environments, where one or more variables are hidden.

---
* Another, more modest, challenge exists in how to choose the optimal policy when no environment is fully observable but the dynamic process for how the states evolve over time is unknown. 
* It may be possible, for simple problems, to reason about how the states evolve, perhaps adding constraints on the state-action space. However,the problem is especially acute in high-dimensional discrete state spaces, arising from, say, discretizing continuous state spaces. 
* Here, it is typically intractable to enumerate all combinations of states and actions and it is hence not possible to exactly solve the optimal control problem. 
* In particular, we will turn to neural networks to approximate an action function known as a “Q-function.” 
* Such an approach is referred to as “Q-Learning” and more recently, with the use of deep learning to approximate the Q-function, is referred to as “Deep Q-Learning.”

---

* To fix ideas, we consider a number of examples to illustrate different aspects of the problem formulation and challenge in applying reinforcement learning. We start with arguably the most famous toy problem used to study stochastic optimal control theory, the “**multi-armed bandit problem**.” This problem is especially helpful in developing our intuition of how an agent must balance the competing goals of exploring different actions versus exploitation of known outcomes.

---

### Example 3 Multi-armed Bandit Problem
* Suppose there is a fixed and finite set of n actions, a.k.a. arms, denoted $A$. Learning proceeds in rounds, indexed by $t = 1,...,T$ . The number of rounds $T $, a.k.a. the time horizon, is fixed and known in advance. In each round, the agent picks an arm at and observes the reward $R_t(a_t)$ for the chosen arm only. For avoidance of doubt, the agent does not observe rewards for other actions that could have been selected. 
* If the goal is to maximize total reward over all rounds, how should the agent choose an arm?

---
* Suppose the rewards Rt are independent and identical random variables with
distribution $ν ∈ [0, 1]^
n$ and mean $μ$. The best action is then the distribution
with the maximum mean $μ^∗$.

* The difference between the player’s accumulated reward and the maximum
the player (a.k.a. the “cumulative regret”) could have obtained had she known
all the parameters is

$$
\bar{R}_T = T\mu^* - \mathbb E \sum_{t\in[T]}R_t
$$

* Intuitively, an agent should pick arms that performed well in the past, yet
the agent needs to ensure that no good option has been missed.

---

* The theoretical origins of reinforcement learning are in stochastic dynamic
programming. In this setting, an agent must make a sequence of decisions under
uncertainty about the reward. 
* If we can characterize this uncertainty with probability distributions, then the problem is typically much easier to solve. We shall assume that you has some familiarity with dynamic programming—the extension to stochastic dynamic programming is a relatively minor conceptual development.
* The following optimal payoff example will    As we follow the mechanics of solving the problem, the example exposes the inherent difficulty of relaxing our assumptions about the distribution of the uncertainty.

---

### Example 4 Uncertain Payoffs

* A strategy seeks to allocate \$600 across 3 markets and is equally profitable
once the position is held, returning 1% of the size of the position over a short
trading horizon $[t,t + 1]$. However, the markets vary in liquidity and there is
a lower probability that the larger orders will be filled over the horizon. The
amount allocated to each market must be either $K = \{100, 200, 300\}$.

![width:800px](Presentation1104\fig7.png)

----

* The optimal allocation problem under uncertainty is a stochastic dynamic programming problem. 
* We can define value functions vi(x) for total allocation amount x for each stage of the problem, corresponding to the markets. 
* We then find the optimal allocation using the backward recursive formulae:
$$
v_3(x)=R_3,\forall x \in K\\
\qquad
v_2(x) = \underset{k \in K}{max}{\{R_2(k)+v_3(x-k)\}},\forall x \in K+200,\\

\qquad
v_1(x) = \underset{k \in K}{max}{\{R_1(k)+v_2(x-k)\}},x = 600,
$$


* The left-hand side of the table below tabulates the values of $R_2 + v_3$
corresponding to the second stage of the backward induction for each pair
$(M_2, M_3)$.

---

![width:1200px](Presentation1104\fig8.png)

* The right-hand side of the above table tabulates the values of $R_1 + v_2$ corresponding to the third and final stage of the backward induction for each tuple $(M_1, M_2^∗, M_3^∗)$.

---
* In the above example, we can see that the allocation $\{200, 200, 200\}$ maximizes $v_1(600) = 4.3$.   
* While this eample is a straightforward application of a Bellman optimality recurrence relation, it provides a glimpse of the types of stochastic dynamic programming problems that can be solved with reinforcement learning.   
* In particular, if the fill probabilities are unknown but must be learned over time by observing the outcome over each period $[t_i, t_i+1)$, then the problem above cannot be solved by just using backward recursion.  

---
* Instead we will move to the framework of reinforcement learning and attempt to learn the best actions given the data.   
* Clearly, in practice, the example is much too simple to be representative of real-world problems in finance—the profits will be unknown and the state space is significantly larger, compounding the need for reinforcement learning.   
* However, it is often very useful to benchmark reinforcement learning on simple stochastic dynamic programming problems with closed-form solutions.

---

* In the previous example, we assumed that the problem was static—the variables
in the problem did not change over time. 
* This is the so-called static allocation problem and is somewhat idealized. 
* Our next example will provide a glimpse of the types of problems that typically arise in optimal portfolio investment where random variables are dynamic. 
* The example is also seated in more classical finance theory, that of a “Markowitz portfolio” in which the investor seeks to maximize a risk-adjusted long-term return and the wealth process is self-financing.

---
### Example 5 Optimal Investment in an Index Portfolio

* Let St be a time-t price of a risky asset such as a sector exchange-traded fund
(ETF). We assume that our setting is discrete time, and we denote different time
steps by integer valued-indices $t = 0,...,T$ , so there are $T + 1$ values on our
discrete-time grid. The discrete-time random evolution of the risky asset $S_t$ is
$$
S_{t+1} = S_t(1+\phi_t)\qquad\qquad\tag{12}
$$

---
$$
S_{t+1} = S_t(1+\phi_t)\qquad\qquad\tag{12}
$$

* where $\phi_t$ is a random variable whose probability distribution may depend on
the current asset value $S_t$ . To ensure non-negativity of prices, we assume that
$\phi_t$ is bounded from below $\phi_t ≥ −1$.

---
* Consider a wealth process $W_t$ of an investor who starts with an initial wealth
$W_0$ = 1 at time $t = 0$ and, at each period $t = 0,...,T − 1$ allocates a fraction
$u_t = u_t(S_t)$ of the total portfolio value to the risky asset, and the remaining
fraction $1 − u_t$ is invested in a risk-free bank account that pays a risk-free
interest rate $r_f = 0$. 
* We will refer to a set of decision variables for all time steps as a policy $\pi:= \{u_t\}^{T-1}_{t=0}$. The wealth process is self-financing and so the
wealth at time $t + 1$ is given by

$$
W_{t+1} =(1-u_t)W_t+u_tW_t(1+\phi_t)\qquad\qquad\tag{13}
$$

This produces the one-step return

$$
r_t={ {W_{t+1}-W_t}\over{W_t} }= u_t{\phi_t}\qquad\qquad\tag{14}
$$

---
* Note this is a random function of the asset price $S_t$ . We define one-step rewards
$R_t$ for $t = 0,...,T − 1$ as risk-adjusted returns

$$
R_t = r_t-\lambda Var[r_t|S_t] = u_t{\phi}_t-\lambda u^2Var[{\phi}_t|S_t],\qquad\qquad\tag{15}
$$

where $λ$ is a risk-aversion parameter.We now consider the problem of
maximization of the following concave function of the control variable $u_t$ :
$$
V^{\pi}(s) = \underset{u_t}{max}\mathbb E\left[\sum_{t_0}^T{R_t}|S_t = s \right] = \underset{u_t}{max}\mathbb E \left[ \sum_{t=0}^T u_t \phi _t -\lambda {u_t}^2Var[\phi _t|S_t]\middle|S_t = s \right] \qquad\tag{16}
$$

* This equation defines an optimal investment problem for T − 1 steps faced
by an investor whose objective is to optimize risk-adjusted returns over each period. 

----
* This optimization problem is equivalent to maximizing the long-run
 returns over the period $[0, T]$. For each $t = T −1, T −2,..., 0,$ the optimality
condition for action ut is now obtained by maximization of $V^π (s)$ with respect
to $u_t$ :

$$
u^* = {{{\mathbb{} E }[\phi _t|S_t]}\over{2 \lambda Var[\phi _t|S_t]}}\qquad\tag{17}
$$

* where we allow for short selling in the ETF $(i.e., u_t < 0)$ and borrowing of
cash $u_t > 1$.

* This is an example of a stochastic optimal control problem for a portfolio that
maximizes its cumulative risk-adjusted return by repeatedly rebalancing between
cash and a risky asset. Such problems can be solved using means of dynamic
programming or reinforcement learning.

