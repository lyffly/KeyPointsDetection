# KeyPointsDetection 关键点识别网络
程序编写与测试：刘云飞

框架：PyTorch 1.3

语言：Python 3.7

在ubuntu 18.03测试通过

#### 0x00 数据准备

标注数据类似下面，512x512 pixel

![](images/data1.png)

![](images\data2.png)

增加上2D高斯  点云 的Heatmap图如下

![](images\card_keypoints.png)

#### 0x01 训练

训练使用Houeglass模型，128*128进行训练

基本结构如下，漏斗式结构

![](images\hour.png)

#### 0x02 结果

下图左侧为检测到的四个点的heatmap，右侧为加上原图的效果，可以看到效果还不错～

![](images\result1.png)



![](images\result2.png)