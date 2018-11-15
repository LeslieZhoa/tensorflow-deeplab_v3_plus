#类别数
num_classes=21
#数据目录
data_dir='./data/VOCdevkit/VOC2012'
#生成tfrecords放置目录
tfrecord_path='./data/tfrecord/'
#训练图片index
train_data_list='./data/train.txt'
#验证图片index
val_data_list='./data/val.txt'
#图片目录
image_data_dir='JPEGImages'
#label目录，每一个像素点即为所分的类别
label_data_dir='SegmentationClassAug'

#模型目录
model_dir='./model'
#是否清除模型目录
clean_model_dir='store_false'
#训练epoch
train_epochs=2
#训练期间的验证次数
epochs_per_eval=1

#tensorboard最大图片展示数
tensorboard_images_max_outputs=6

#批次设置
batch_size=4
#学习率衰减策略
learning_rate_policy='poly'
#学习率衰减最大次数
max_iter=30000

#重载的结构
base_architecture='resnet_v2_101'
#预训练模型位置
pre_trained_model='./resnet_v2_101/resnet_v2_101.ckpt'
#模型encoder输入与输出比例
output_stride=16
#是否更新BN参数
freeze_batch_norm='store_true'
#起始学习率
initial_learning_rate=7e-3
#终止学习率
end_learning_rate=1e-6
#global_step初始值
initial_global_step=0
#正则化权重
weight_decay=2e-4

debug=None

#测试图片地址
pictue='./picture/'
#测试图片输出地址
output='./output/'
#测试输入，若为1则输入图片，为2输入是摄像头
test_mode='1'

