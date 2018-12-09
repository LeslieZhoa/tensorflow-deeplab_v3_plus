# tensorflow-deeplab_v3_plus
参考[rishizek](https://github.com/rishizek/tensorflow-deeplab-v3-plus)的代码进行中文注释，并按照自己风格重新编写代码，对ASPP加入里BN层，支持摄像头。<br>
## deeplab_v3_plus简介
图像分割是主要功能是将输入图片的每个像素都分好类别，也相当于分类过程。举例来说就是将大小为[h,w,c]的图像输出成[h,w,1]，每个像素值代表一个类别。<br>
deeplab_v3+可以参考论文[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)。它的结构图如下：<br>
![](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/output/model1.png)<br>
![](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/output/model2.png)<br>
下面对模型进行简要分析<br>
该模型属于encoder-decoder模型，encoder-decoder常用于自然语言处理中，在图像分割中[U-net](https://arxiv.org/pdf/1505.04597.pdf)也是十分典型的encoder-decoder模型，大体结构如下：<br>
![](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/output/UNET.png)<br>
就是将图片通过卷积尺寸变小再通过上采样将尺寸还原。<br><br>
deeplab_v3+是将encoder-decoder和ASPP相结合，encoder-decoder会获取更多边界信息，ASPP可获取更多特征信息。encoder网络使用resnet101或 Xception,本代码中使用的是resnet101。<br><br>
采用预训练的resnet的某一节点A来获取图像信息，再加入到ASPP中。ASPP就是不同膨胀率的空洞卷积和全局池化上采样后的输出concat在一起，作为encoder输出部分。<br><br>
空洞卷积可以理解为一个大卷积中间权重值都为0,举例说明，一个3x3的卷积，如果膨胀率是1就是正常卷积，如果膨胀率是2,就是空洞卷积，相当于把3x3的卷积每个值的右方和下方加一行或列都置0。变换之后的空洞矩阵大小变为6x6。空洞矩阵论文中说可以提取更密集的特征，捕获多尺度信息，相比于卷积和池化会减少信息丢失。全局池化就是将输入[h,w,c]池化成[1,1,c]。<br><br>
decoder部分选取resnet中A节点之前的B节点，再将encoder的输出上采样成B的大小然后concat，做一些卷积和上采样就得到最终输出。<br><br>
由于可以看成分类问题，该模型的损失函数也是交叉熵函数。模型具体实现可以参考代码<br>
## 模型训练
### 环境要求
ubuntu=16.04<br>
tensorflow=1.4.1<br>
opencv=3.4.1<br>
windows下可以进行测试<br>
### 下载数据集
将[VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)解压到data目录下,下载DrSleep提供的[SegmentationClassAug文件](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)解压，里面有_MACOSX和SegmentationClassAug两个文件夹，将里面的SegmentationClassAug放置到./data//VOCdevkit/VOC2012/下，里面的图片是shape为[h,w,1]每一个像素值都对应类别的label。<br><br>
将[txt](https://share.weiyun.com/5pjrMSV)文件解压，里面的.txt文件放置到./data下，用于分化训练集和验证集。<br><br>
将[restnet预训练数据](https://www.jianguoyun.com/p/DZeHNHwQv7mYBxiBsIwB)解压放置在该模型的根目录下。<br><br>
如果需要模型预训练数据可以将我训练的[权重数据](https://share.weiyun.com/5lGifzi)解压,将里面的ckpt等文件放置到./model下。<br><br>
### 代码介绍
data放置VOC数据和数据处理生成的record文件<br><br>
model放置训练生成的模型和graph<br><br>
output放置测试图片生成的分割图像<br><br>
picture放置测试用例,我的来源于百度图片<br><br>
utils包含配置文件config.py,数据处理文件dataset_util.py,preprocessing.py和模型文件deeplab_model.py<br><br>
test.py是测试文件支持摄像头<br><br>
tfrecord.py是将处理完的数据生成record文件<br><br>
train.py是训练文件<br>
### 运行
手动配置config.py的信息或选择默认<br><br>
若要训练：<br>
运行python tfrecord.py生成record文件<br>
运行python train.py训练。  对于计算机gpu选择需修改[这里代码](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/train.py#L38)<br><br>
若要测试：<br>
运行python test.py<br><br>
## 一些疑问
我的电脑配置是1080Ti但总是运行运行就溢出，我尝试用tf.contrib.distribute.mirroredstrategy多gpu并行，但tensorflow版本要1.8,当我更新完，发现input_fn要是data格式，我失败了。<br>
如果有并行gpu的建议或者代码的指正请给我留言<br>
## 结果展示
![](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/output/2007_000346.jpg)
![](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/output/2007_000464.jpg)
![](https://github.com/LeslieZhoa/tensorflow-deeplab_v3_plus/blob/master/output/2007_000243.jpg)

