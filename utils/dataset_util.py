
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


#tfrecords转换的各种类型
def int_64_feature(value):
    return tf.train.Feature(int_64_feature=tf.train.Int64List(value=[value]))
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# In[3]:


def read_examples_list(path):
    '''返回所有图片的index'''
    with tf.gfile.GFile(path) as f:
        lines=f.readlines()
    return [line.strip().split(' ')[0] for line in lines]
    

