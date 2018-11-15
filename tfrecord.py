
# coding: utf-8

# In[1]:


import io
import os
from PIL import Image
import tensorflow as tf
from utils import config as Config
from utils import dataset_util


# In[2]:


def main():
    '''生成tfrecords主程序
    '''
    if not os.path.exists(Config.tfrecord_path):
        os.makedirs(Config.tfrecord_path)
    #相当于print
    tf.logging.info('读取数据')
    
    image_dir=os.path.join(Config.data_dir,Config.image_data_dir)
    label_dir=os.path.join(Config.data_dir,Config.label_data_dir)
    
    if not os.path.isdir(label_dir):
        raise ValueError('数据缺少，去下载')
    #获取训练和验证图片的index
    train_examples=dataset_util.read_examples_list(Config.train_data_list)
    val_examples=dataset_util.read_examples_list(Config.val_data_list)
    
    #训练验证tfrecord存储地址
    train_output_path=os.path.join(Config.tfrecord_path,'train.record')
    val_output_path=os.path.join(Config.tfrecord_path,'val.record')
    
    #生成tfrecord
    create_record(train_output_path,image_dir,label_dir,train_examples)
    create_record(val_output_path,image_dir,label_dir,val_examples)


# In[3]:


def create_record(output_filename,image_dir,label_dir,examples):
    '''将图片生成tfrecord
    参数：
      output_filename:输出地址
      image_dir:图片地址
      label_dir:label地址
      examples：图片的index名字
      '''
    writer=tf.python_io.TFRecordWriter(output_filename)
    for idx,example in enumerate(examples):
        if idx % 500 ==0:
            #将生成第几张图片信息输出
            tf.logging.info('On image %d of %d',idx,len(examples))
        image_path=os.path.join(image_dir,example+'.jpg')
        label_path=os.path.join(label_dir,example+'.png')
        
        if not os.path.exists(image_path):
            tf.logging.warning('没有该图片: ',image_path)
            continue
        elif not os.path.exists(label_path):
            tf.logging.warning('没找着label文件： ',label_path)
            continue
        try:
            #转换格式
            
            tf_example=dict_to_tf_example(image_path,label_path)
           
            writer.write(tf_example.SerializeToString())
        except ValueError:
            tf.logging.warning('无效的example： %s, 忽略',example)
    writer.close()


# In[4]:


def dict_to_tf_example(image_path,label_path):
    '''格式转换成tfrecord
    参数：
      image_path:输入图片地址
      label_path:输出label地址
      '''
    with tf.gfile.GFile(image_path,'rb') as f:
        encoder_jpg=f.read()
    encoder_jpg_io=io.BytesIO(encoder_jpg)
    image=Image.open(encoder_jpg_io)
  
    if image.format !='JPEG':
        tf.logging.info('输入图片格式错误')
        raise ValueError('输入图片格式错误')
    
    with tf.gfile.GFile(label_path,'rb') as f:
        encoder_label=f.read()
    encoder_label_io=io.BytesIO(encoder_label)
    label=Image.open(encoder_label_io)
    
    if label.format !='PNG':
        tf.logging.info('label图片格式错误')
        raise ValueError('label图片格式错误')
    
    if image.size!=label.size:
        tf.logging.info('输入输出没对上')
        raise ValueError('输入输出没对上')
   
    example=tf.train.Example(features=tf.train.Features(feature={
        'image':dataset_util.bytes_feature(encoder_jpg),
        'label':dataset_util.bytes_feature(encoder_label)}))
    return example
    


# In[5]:


if __name__=='__main__':
    #为将要被记录的的东西（日志）设置开始入口
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

