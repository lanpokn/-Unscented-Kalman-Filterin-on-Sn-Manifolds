# 算法大纲梳理

由于原论文中公式指引过于杂乱，有必要重新梳理一下算法

在算法之前，先给出process function(f)和observation function的定义如下：

![image-20240705000520067](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000520067.png)

![image-20240705000529945](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000529945.png)

注意xt经过了exp，因此是在流形上的东西，yt是RN上的东西。当流形是球时，exp的定义如下：

![image-20240705000702511](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000702511.png)

对应的LOG为：

![image-20240705000735292](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000735292.png)

注意到该算法是一个迭代形的算法，对于输入的x的均值和方差，输出下一轮的均值和方差即可

![image-20240705000237980](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000237980.png)、首先要根据输入的均值和方差，利用process function，计算自然运动后的均值和方差估计值，这两个公式分别为：

![image-20240705000438911](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000438911.png)

![image-20240705000412660](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705000412660.png)

我在代码中部署的也是这个。但这个存在严重的问题！具体来讲，f是流形到流形的，因此这样做完x都不一定在流形上了，因此要换成流形版本（TODO)：

![image-20240705001233965](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705001233965.png)

![image-20240705001312733](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705001312733.png)

(TODO)注意其中带下标的不是原本sigma point,而是**变化到流形空间过后**的，一定要密切注意在正切空间还是流形空间，因此unsenet函数写多个版本才是合理的;

![image-20240705002602747](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705002602747.png)

cov的计算是自然的（前提LOG要算对），但是E需要推导，这个可以直接求助gpt，让他生成一个函数，计算一堆单位球面上点的球面上均值，自然是可以完成的。目前代码没有实现这个问题

其中权重和sigma点的计算公式为（TODO，没检查过，要检查）：

![image-20240705001524831](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705001524831.png)

![image-20240705001548971](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705001548971.png)

![image-20240705001628609](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705001628609.png)

这就是要计算yt_hat, Pyy, Pxy，注意着依然没有涉及到新的观测信息yt，只是利用观测方程做一些预估而已.他们的计算公式分别为：

（该问题下观测方程在Rn上，只看简化版本即可）

![image-20240705002051280](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705002051280.png)

（因为观测方程在Rn上，不是球面上加噪，所以我可以这么写，不过一定要记住观测值和状态值不在同一个流形空间中）

![image-20240705002150758](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705002150758.png)

![image-20240705002206563](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705002206563.png)

（TODO)Pyy就是简单的Rn方差但是带下标，Pxy有的有下标有的没有，这意味着我目前代码还是有问题，需要改这个下标的问题

**经验教训：理论文章绝不能有任何地方是觉得难以定义无法理解的**

![image-20240705003309562](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705003309562.png)

注意它后边又不说是24了，所以完整公式如下：

卡尔曼增益：

![image-20240705003425084](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705003425084.png)

方差更新：

![image-20240705003447422](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705003447422.png)

均值

![image-20240705003738616](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705003738616.png)

显然这东西（Pt)和xt都是正切空间上的，xt需要换回到流形上（其实就是把向量看作相对于某一点切向量，然后投影回去，当然实际是用速度和测地线描述的），但问题在于，根据前边定义，方差本就是用正切空间中表示的，似乎不需要投影回去？

![image-20240705004041381](C:\Users\hhq\AppData\Roaming\Typora\typora-user-images\image-20240705004041381.png)

第四步有点迷惑，xt没什么好说的，似乎Pt是要从一个点的正切空间中切换到另一个点的正切空间中。这是因为流形中方差的定义居然是和选取点有关的，很不像传统方差，因此需要转换

TODO：完成这个平行传递，gpt的大概率不靠谱，文章中虽然讲了，但大概率不靠谱



综上，主要的问题有：E[f(x)]均值计算错误，sigma下标理解错误，平行传递缺失

次要问题有：检查代码正确性



TODO:必须手动保证P都是与所选点相切的，否则sigma point会生成到切平面之外

目前算法问题在于，P只有取局部参考系才是有效的，但是我实现不出来。

没想到这么难，我应该让其在局部生成，再转回去



问题分析：首先，如果x是n维欧拉，P应该是n-1维度的，肯定有一维没有意义

其次，如果想用局部表示，那就必须解决表示混乱的问题。

应该先有一个函数，对于任意x作为法向量，生成一个唯一的坐标系表示，且坐标系之间平滑变化。

然后，任意P实际上坐落于正切空间中。P比x小一维，永远都只是正切空间中的？

那好像也不行，根据公式来看P和x是同维度的。

能否对exp动手脚？保证v在正切空间中？

目前是强行让Exp对任意向量都能投影了，其实就是只看正交部分，但这样搞P还是有问题，Pyy,问题最小，Pxy经常崩溃

但起码现在运行一步的结果可以交差了，起码能跑，结果没拐弯，但由于方差错误，跑不远。。。。

第三张图不用怕，是很容易分析的

好吧，当我没说，完全无法交差
如果方差固定了能跑通，那就核心问题聚焦在方差更新上了



Pxy很可能是2x3维的，也许不重要？

总之，我需要时刻保证sigma point生成在切平面内。其实坐标系根本无所谓，我还是只记录全局P，每次需要局部计算时，弄一个临时的坐标系转换，计算出临时的P以及临时的切平面上的sigma point，再把sigma point转换回全局表示即可。 只要我能随时保证sigma point时在切平面内的，P用全局表示反而正确，因为相当于换点时平行传递，就只是把P进行一下旋转即可，便于表示。



所以关键在于：sigma point前后都要进行坐标系转换，在与x正交的平面上计算，应该就可以了。

实在不行先把粒子滤波实现一下吧。。总之最后剩两天时，以完成任务为目标，这周还是竭尽全力。



总之原文章里倾向把P降维表示，要清楚这一点。只要我能想明白任意维度球的坐标系问题，降维表示未尝不可。但我还是怀疑怎么表示都无所谓。



TODO:

先别急，还是先把log和exp测试明白了再说，我现在Exp是假定正切空间由全局坐标系表示，那么LOG返回的是不是全局坐标系呢？这需要测试的，想把log和exp玩明白再说，实在不行画图提醒自己。我终于明白为什么五年后还有论文专门给这算法写代码了，理论分析的基础上，代码实现方式完全没确定。。。确定好这个了，再去改sigmapoint的逻辑，就差不多了。

惊天发现，锁定了都不行，方差照样是非正定？什么鬼？原来pred那里就已经崩了。但是都true也能来个false,但pred也是有概率崩盘，再思考吧

稳定性很强，如果真正锁好方差那么无论什么维度，都是正结果。因此还是要有自信的，搞不好就是log把我坑了。平行也好办，有了局部坐标系，本质上是APB，先转局部，然后把这个局部当成另一个的局部，再转到全局