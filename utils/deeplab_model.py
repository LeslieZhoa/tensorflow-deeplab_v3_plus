
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
slim=tf.contrib.slim
from utils import preprocessing
_BATH_NORM_DECAY=0.9997
_WEIGHT_DECAY=5e-4


# In[10]:


def model_fn(features,labels,mode,params):
    '''对于estimator的模型接口
    参数：
      features:输入特征
      labels:真实label
      mode:模型模式
      params:模型运行相关参数
    返回值：
      模型接口形式
      '''
    if isinstance(features,dict):
        features=features['feature']
    #图像加上均值，以便显示
    images=tf.cast(tf.map_fn(preprocessing.mean_image_addition,features),
                  tf.uint8)
    network=model_generator(params['num_classes'],
                           params['output_stride'],
                           params['base_architecture'],
                           params['pre_trained_model'],
                           params['batch_norm_decay'])
    logits=network(features,mode==tf.estimator.ModeKeys.TRAIN)
    #预测类别shape[batch,h,w,1]
    pred_classes=tf.expand_dims(tf.argmax(logits,axis=3,output_type=tf.int32),axis=3)
    #图片上色形式shape[batch,h,w,3]
    pred_decoded_labels=tf.py_func(preprocessing.decode_labels,
                                  [pred_classes,params['batch_size'],params['num_classes']],
                                  tf.uint8)
    
    predictions={
        'classes':pred_classes,
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor'),
        'decoded_labels':pred_decoded_labels
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        #模式为预测，将decoded_labels删掉
        predictions_without_decoded_labels=predictions.copy()
        del predictions_without_decoded_labels['decoded_labels']
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'preds':tf.estimator.export.PredictOutput(
                    predictions_without_decoded_labels)
                })
    #为真实label上色
    gt_decoded_labels=tf.py_func(preprocessing.decode_labels,
                                [labels,params['batch_size'],params['num_classes']],tf.uint8)
    
    labels=tf.squeeze(labels,axis=3)#[batch,h,w]
    logits_by_num_classes=tf.reshape(logits,[-1,params['num_classes']])#[-1,21]
    labels_flat=tf.reshape(labels,[-1,])#[-1]
    #有类别的像素遮罩
    valid_indices=tf.to_int32(labels_flat<=params['num_classes']-1)
    #除去不明类别的预测和真实值
    valid_logits=tf.dynamic_partition(logits_by_num_classes,valid_indices,num_partitions=2)[1]#[-1,num_classes]
    valid_labels=tf.dynamic_partition(labels_flat,valid_indices,num_partitions=2)[1]#[-1]
    
    pred_flat=tf.reshape(pred_classes,[-1,])#[-1]
    valid_preds=tf.dynamic_partition(pred_flat,valid_indices,num_partitions=2)[1]#[-1]
    #列代表真实值，行代表预测值的混淆矩阵
    confusion_matrix=tf.confusion_matrix(valid_labels,valid_preds,num_classes=params['num_classes'])
    predictions['valid_preds']=valid_preds
    predictions['valid_labels']=valid_labels
    predictions['confusion_maxtrix']=confusion_matrix
    
    #损失函数为交叉熵
    cross_entropy=tf.losses.sparse_softmax_cross_entropy(
    logits=valid_logits,labels=valid_labels)
    
    #记录信息
    tf.identity(cross_entropy,name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
    
    #训不训练BN里的数值
    if not params['freeze_batch_norm']:
        train_var_list=[v for v in tf.trainable_variables()]
    else:
        train_var_list=[v for v in tf.trainable_variables()
                       if 'beta' not in v.name and 'gamma' not in v.name]
    #加上正则计算总损失
    with tf.variable_scope('total_loss'):
        loss=cross_entropy+params.get('weight_decay',_WEIGHT_DECAY)*tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
        
    #加入图片到tensrboard
    if mode==tf.estimator.ModeKeys.TRAIN:
        tf.summary.image('image',
                        tf.concat(axis=2,values=[images,gt_decoded_labels,pred_decoded_labels]),
                        max_outputs=params['tensorboard_images_max_outputs'])
        global_step=tf.train.get_or_create_global_step()
        #选择学习率衰减模式
        if params['learning_rate_policy']=='piecewise':
            initial_learning_rate=0.1*params['batch_size']/128
            #每一个epoch有几个batch
            batches_per_epoch=params['num_train']/params['batch_size']
            boundaries=[int(batches_per_epoch*epoch) for epoch in [100,150,200]]
            values=[initial_learning_rate*decay for decay in [1,0.1,0.01,0.001]]
            learning_rate=tf.train.piecewise_constant(
            tf.cast(global_step,tf.int32),boundaries,values)
        elif params['learning_rate_policy']=='poly':
            learning_rate=tf.train.polynomial_decay(
            params['initial_learning_rate'],
            tf.cast(global_step,tf.int32)-params['initial_global_step'],
            params['max_iter'],params['end_learning_rate'],power=params['power'])
        else:
            raise ValueError('选择一个学习率模型啊')
        tf.identity(learning_rate,name='learning_rate')
        tf.summary.scalar('learning_rate',learning_rate)
        
        tf.identity(global_step,name='global_step')
        tf.summary.scalar('global_step',global_step)
        optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=params['momentum'])
        #BN需相关更新
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op=optimizer.minimize(loss,global_step,var_list=train_var_list)
    else:
        train_op=None
    
    #准确率和平均iou计算
    accuracy=tf.metrics.accuracy(valid_labels,valid_preds)
    mean_iou=tf.metrics.mean_iou(valid_labels,valid_preds,params['num_classes'])
    metrics={'px_accuracy':accuracy,'mean_iou':mean_iou}
    
    tf.identity(accuracy[1],name='train_px_accuracy')
    tf.summary.scalar('train_px_accuracy',accuracy[1])
    
    def compute_mean_iou(total_cm,name='mean_iou'):
        '''计算平均iou
        参数：
          total_cm：混淆矩阵
        返回值：平均iou
        '''
        #分别计算按行按列总数，shape[num_classes]
        sum_over_row=tf.to_float(tf.reduce_sum(total_cm,0))
        sum_over_col=tf.to_float(tf.reduce_sum(total_cm,1))
        #计算对角线即预测正确总数
        cm_diag=tf.to_float(tf.diag_part(total_cm))
        #分母，shape[num_classes]代表每一个类别
        denominator=sum_over_row+sum_over_col-cm_diag
        
        #计算多少类别有预测值
        num_valid_entries=tf.reduce_sum(tf.cast(
            tf.not_equal(denominator,0),dtype=tf.float32))
        #避免分母为0
        denominator=tf.where(tf.greater(
            denominator,0),denominator,
            tf.ones_like(denominator))
        iou=tf.div(cm_diag,denominator)
        
        for i in range(params['num_classes']):
            tf.identity(iou[i],name='train_iou_class{}'.format(i))
            tf.summary.scalar('train_iou_class{}'.format(i),iou[i])
        result=tf.where(
            tf.greater(num_valid_entries,0),
            tf.reduce_sum(iou,name=name)/num_valid_entries,
            0)
        return result
    train_mean_iou=compute_mean_iou(mean_iou[1])
    tf.identity(train_mean_iou,name='train_mean_iou')
    tf.summary.scalar('train_mean_iou',train_mean_iou)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


# In[1]:


def model_generator(num_classes,output_stride,
                   base_architecture,
                   pre_trained_model,
                   batch_norm_decay,
                   data_format='channels_last'):
    '''模型主程序
    参数：
      num_classes：类别
      output_stride:resnet的步长还和空洞卷积膨胀系数有关，若为16,系数为[6,12,18],为8,系数翻倍
      base_architecture:resnet的重载模型
      pre_trained_model:预训练模型目录
      batch_norm_decay:BN层的系数
      data_format:输入图片的格式，RGB通道在最前还是最后
    返回值：
      返回预测值shape[batch,h,w,num_classes]
      '''
    if data_format is None:
        pass
    if batch_norm_decay is None:
        batch_norm_decay=_BATH_NORM_DECAY
    if base_architecture not in ['resnet_v2_50','resnet_v2_101']:
        raise ValueError('重载模型没整对')
    if base_architecture =='resnet_v2_50':
        base_model=resnet_v2.resnet_v2_50
    else:
        base_model=resnet_v2.resnet_v2_101
    #建立模型
    def model(inputs,is_training):
        #统一输入格式为RGB通道放最后
        if data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])
       
        #重载resnet
        with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits,end_points=base_model(inputs,
                                        num_classes=None,
                                        is_training=is_training,
                                        global_pool=False,
                                        output_stride=output_stride)
        if is_training:
            #重载权重
            exclude=[base_architecture+'/logits','global_step']
            variables_to_restore=slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(pre_trained_model,
                                         {v.name.split(':')[0]: v for v in variables_to_restore})
        inputs_size=tf.shape(inputs)[1:3]
        #取一个resnet网络节点
        net=end_points[base_architecture+'/block4']
        #resnet节点经过ASPP作为编码输出
        encoder_output=atrous_spatial_pyramid_pooling(net,output_stride,batch_norm_decay,is_training)
        
        #解码将图片恢复原来大小
        with tf.variable_scope('decoder'):
            with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
#                 with slim.arg_scope([slim.conv2d],
#                                     weights_initializer=slim.xavier_initializer(),

#                                     normalizer_fn=slim.batch_norm,
#                                     normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
                with tf.variable_scope('low_level_features'):
                    #又搞来一个节点
                    low_level_features=end_points[base_architecture+'/block1/unit_3/bottleneck_v2/conv1']
                    low_level_features=slim.conv2d(low_level_features,48,[1,1],stride=1,scope='conv_1x1')
                    low_level_features_size=tf.shape(low_level_features)[1:3]

                with tf.variable_scope('upsampling_logits'):
                    #上采样成输入大小
                    net=tf.image.resize_bilinear(encoder_output,low_level_features_size,name='upsample_1')
                    net=tf.concat([net,low_level_features],axis=3,name='concat')
                    net=slim.conv2d(net,256,[3,3],stride=1,scope='conv_3x3_1')
                    net=slim.conv2d(net,256,[3,3],stride=1,scope='conv_3x3_2')
                    net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='conv_1x1')
                    logits=tf.image.resize_bilinear(net,inputs_size,name='upsample_2')
        return logits
    return model


# In[7]:


def atrous_spatial_pyramid_pooling(inputs,output_stride,
                                  batch_norm_decay,is_training,depth=256):
    '''实现ASPP
    参数：
      inputs：输入四维向量
      output_stride：决定空洞卷积膨胀率
      batch_norm_decay:同上函数
      is_training:是否训练
      depth:输出通道数
    返回值：
      ASPP后的输出
      '''
    with tf.variable_scope('aspp'):
        if output_stride not in [8,16]:
            raise ValueError('out_stride整错了')
        #膨胀率
        atrous_rates=[6,12,18]
        if output_stride ==8:
            atrous_rates=[2*rate for rate in atrous_rates]
        with slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=slim.xavier_initializer(),
                                
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
                inputs_size=tf.shape(inputs)[1:3]
                #slim.conv2d默认激活函数为relu,padding=SAME
                conv_1x1=slim.conv2d(inputs,depth,[1,1],stride=1,scope='conv_1x1')
                #空洞卷积rate不为1
                conv_3x3_1=slim.conv2d(inputs,depth,[3,3],stride=1,rate=atrous_rates[0],scope='conv_3x3_1')
                conv_3x3_2=slim.conv2d(inputs,depth,[3,3],stride=1,rate=atrous_rates[1],scope='conv_3x3_2')
                conv_3x3_3=slim.conv2d(inputs,depth,[3,3],stride=1,rate=atrous_rates[2],scope='conv_3x3_3')
                with tf.variable_scope('image_level_features'):
                    #池化
                    image_level_features=tf.reduce_mean(inputs,axis=[1,2],keep_dims=True,name='global_average_pooling')
                    image_level_features=slim.conv2d(image_level_features,depth,[1,1],stride=1,scope='conv_1x1')
                    #双线性插值
                    image_level_features=tf.image.resize_bilinear(image_level_features,inputs_size,name='upsample')
                net=tf.concat([conv_1x1,conv_3x3_1,conv_3x3_2,conv_3x3_3,image_level_features],axis=3,name='concat')
                return net

