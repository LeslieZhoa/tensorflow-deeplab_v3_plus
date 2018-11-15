
# coding: utf-8

# In[1]:


''' 主要进行相关数据预处理'''
from PIL import  Image
import numpy as np
import tensorflow as tf

#三色通道的平均值
_R_MEAN=123.68
_G_MEAN=116.78
_B_MEAN=103.94

#主要为各分类上色
label_colors=[(0,0,0),#0=背景
              #1=飞机，  2=自行车，  3=鸟，      4=船，     5=瓶子
              (128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),
              #6=公交车，    7=小汽车，    8=猫，    9=椅子，    10=牛
              (0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0),
              #11=晚饭桌，   12=狗，    13=马，     14=摩托车，     15=人
              (192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128),
              #16=盆栽， 17=羊，    18=沙发，   19=火车，   20=电视或显示屏
              (0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]


# In[3]:


def decode_labels(mask,num_image=1,num_classes=21):
    '''给图片上色
      参数：
        mask：shape是[batch,h,w,1]像素值为每一个像素点的类别
        num_image：每次处理图片的长数
        num_classes:分类类别数
      返回值：
        返回被上色的分割图像
        '''
    n,h,w,c=mask.shape
    assert (n>=num_image),'num_image %d 不能比批次 %d 大'                            %(n,num_image)
    outputs=np.zeros((num_image,h,w,3),dtype=np.uint8)
    for i in range(num_image):
        #定义一个长宽为h,w的rgb图像
        img=Image.new('RGB',(len(mask[i,0]),len(mask[i])))
        pixels=img.load()
        for j_,j in enumerate(mask[i,:,:,0]):
            for k_,k in enumerate(j):
                #如果类别在区间内，给图片上色
                if k<num_classes:
                    pixels[k_,j_]=label_colors[k]
        outputs[i]=np.array(img)
    return outputs


# In[4]:


def mean_image_addition(image,means=(_R_MEAN,_G_MEAN,_B_MEAN)):
    '''为图像每一通道增加平均值
    参数：
      image：经减去均值的图像shape[h,w,c]
      means：每一通道均值
    返回值：加上均值后的图像
    '''
    if image.get_shape().ndims!=3:
        raise ValueError('图像不对')
    num_channels=image.get_shape().as_list()[-1]
    if len(means)!=num_channels:
        raise ValueError('均值不匹配')
    #将image在第三通道分割成rgb三块
    channels=tf.split(axis=2,num_or_size_splits=num_channels,value=image)
    #每一通道分别加均值
    for i in range(num_channels):
        channels[i]+=means[i]
    #将三通道再组合在一起
    return tf.concat(axis=2,values=channels)
      


# In[6]:


def mean_image_subtraction(image,means=(_R_MEAN,_G_MEAN,_B_MEAN)):
    '''图像减去均值作为输入
    参数：
      image：原始图像[h,w,c]
      means:均值
    返回值：
      减去均值后的图像用作输入
    '''
    if image.get_shape().ndims!=3:
        raise ValueError('图像不对')
    num_channels=image.get_shape().as_list()[-1]
    if len(means)!=num_channels:
        raise ValueError('均值不匹配')
    #将image在第三通道分割成rgb三块
    channels=tf.split(axis=2,num_or_size_splits=num_channels,value=image)
    #每一通道分别减均值
    for i in range(num_channels):
        channels[i]-=means[i]
    #将三通道再组合在一起
    return tf.concat(axis=2,values=channels)


# In[8]:


def random_rescale_image_and_label(image,label,min_scale,max_scale):
    '''随机放大缩小图像
    参数：
      image：输入图像[h,w,c]
      label:分割后的输出图像[h,w,1]
      min_scale,max_scale:尺度改变最小最大值
    返回值：
      改变尺度的image和label
      '''
    if min_scale<=0:
        raise ValueError('最小尺度一定要大于0')
    elif max_scale<=0:
        raise ValueError('最大尺度一定要大于0')
    elif min_scale>=max_scale:
        raise ValueError('尺度大小搞错了')
    shape=tf.shape(image)
    height=tf.to_float(shape[0])
    width=tf.to_float(shape[1])
    #生成随机尺度
    scale=tf.random_uniform([],minval=min_scale,maxval=max_scale,dtype=tf.float32)
    
    new_height=tf.to_int32(height*scale)
    new_width=tf.to_int32(width*scale)
    #双线性插值
    image=tf.image.resize_images(image,[new_height,new_width],
                                method=tf.image.ResizeMethod.BILINEAR)
    #最近邻
    label=tf.image.resize_images(label,[new_height,new_width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image,label


# In[9]:


def random_crop_or_pad_image_and_label(image,label,crop_height,crop_width,ignore_label):
    '''随机裁剪填补图像
    参数：
      image:输入图像[h,w,c]
      label:输出label[h,w,1]
      crop_height,crop_width:新图像尺寸
      ignore_label:被忽略的类别
    返回值：
       处理后的image，label
       '''
    #因为0填充所以要把0减去，否则填充之后原来的0就变成1了
    label=label-ignore_label
    label=tf.to_float(label)
    shape=tf.shape(image)
    height=shape[0]
    width=shape[1]
    image_and_label=tf.concat([image,label],axis=2)
    image_and_label_pad=tf.image.pad_to_bounding_box(
            image_and_label,0,0,
            tf.maximum(crop_height,height),
            tf.maximum(crop_width,width))
    image_and_label_crop=tf.random_crop(
        image_and_label_pad,[crop_height,crop_width,4])
    image_crop=image_and_label_crop[:,:,:3]
    label_crop=image_and_label_crop[:,:,3:]
    label_crop+=ignore_label
    label_crop=tf.to_int32(label_crop)
    return image_crop,label_crop


# In[10]:


def random_filp_left_right_image_and_label(image,label):
    '''随机左右翻转图像
    参数：
      image：输入图像[h,w,c]
      label:输出label[h,w,1]
    返回值：
      处理后的image，label
      '''
    uniform_random=tf.random_uniform([],0,1.0)
    #对比阈值决定翻转
    mirror_cond=tf.less(uniform_random,0.5)
    #tf.cond是判断语句依据概率来翻转
    image=tf.cond(mirror_cond,lambda: tf.reverse(image,[1]),lambda:image)
    label=tf.cond(mirror_cond,lambda:tf.reverse(label,[1]),lambda:label)
    return image,label
    


# In[11]:


def eval_input_fn(image_filenames,label_filenames=None,batch_size=1):
    '''将图像文件夹处理成模型接收data格式
    参数：
      image_filenames:图片目录
      label_filenames:测试数据没有label
      把batch_size:测试默认batch为1
    返回值：
      data形式的数据包含image和label
      '''
    #读取文件中的图片
    def _parse_function(filename,is_label):
        #is_label对于测试数据为None
        if not is_label:
            image_filename,label_filename=filename,None
        else :
            image_filename,label_filename=filename
        image_string=tf.read_file(image_filename)
        image=tf.image.decode_image(image_string)
        image=tf.to_float(tf.image.convert_image_dtype(image,dtype=tf.uint8))
        image.set_shape([None,None,3])
        image=mean_image_subtraction(image)
        if not is_label:
            return image
        else:
            label_string = tf.read_file(label_filename)
            label = tf.image.decode_image(label_string)
            label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
            label.set_shape([None, None, 1])
            return image,label
    if label_filenames is None:
        input_filenames=image_filenames
    else:
        input_filenames=(image_filenames,label_filenames)
    #生成data格式
    dataset=tf.data.Dataset.from_tensor_slices(input_filenames)
    if label_filenames is None:
        dataset=dataset.map(lambda x: _parse_function(x,False))
    else:
        dataset=dataset.map(lambda x,y:_parse_function((x,y),True))
    dataset=dataset.prefetch(batch_size)#和batch一起用加快处理速度
    dataset=dataset.batch(batch_size)
    #生成迭代器
    iterator=dataset.make_one_shot_iterator()
    if label_filenames is None:
        images=iterator.get_next()
        labels=None
    else:
        images,labels=iterator.get_next()
    return images,labels

