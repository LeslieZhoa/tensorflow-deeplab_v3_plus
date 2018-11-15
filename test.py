
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from utils import config as FLAGS
from utils import deeplab_model,preprocessing
import numpy as np
import cv2


# In[2]:


def main():
    image=tf.placeholder(tf.float32,[None,None,3])
    inputs=preprocessing.mean_image_subtraction(image)
    inputs=tf.expand_dims(inputs,axis=0)
    model=deeplab_model.model_generator(FLAGS.num_classes,
                                       FLAGS.output_stride,
                                       FLAGS.base_architecture,
                                       FLAGS.pre_trained_model,
                                       None,)
    logits=model(inputs,False)
    
     #预测类别shape[batch,h,w,1]
    pred_classes=tf.expand_dims(tf.argmax(logits,axis=3,output_type=tf.int32),axis=3)
    #图片上色形式shape[batch,h,w,3]
    pred_decoded_labels=tf.py_func(preprocessing.decode_labels,
                                  [pred_classes,1,FLAGS.num_classes],
                                  tf.uint8)
    pred_decoded_labels=tf.squeeze(pred_decoded_labels)
    saver=tf.train.Saver()
    sess=tf.Session()
    model_file=tf.train.latest_checkpoint(FLAGS.model_dir)
    saver.restore(sess,model_file)
    if FLAGS.test_mode=='1':
        for filename in os.listdir(FLAGS.pictue):
            x=cv2.imread(FLAGS.pictue+filename)
            x1=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
            out=sess.run(pred_decoded_labels,feed_dict={image:x1})
            out=cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
            out=np.concatenate([x, out], axis=1)
            cv2.imshow('im',out)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:        
                cv2.imwrite(FLAGS.output + filename,out)
        cv2.destroyAllWindows()
        
    if FLAGS.test_mode=='2':
        cap=cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(FLAGS.output+'out.mp4' ,fourcc,10,(1280,480))
        while True:
            ret,frame = cap.read()
            if ret == True:
                frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                result=sess.run(pred_decoded_labels,feed_dict={image:frame1})
                result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
                result1=np.concatenate([frame, result], axis=1)
                a = out.write(result1)
                cv2.imshow("result", result1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    sess.close()
    


# In[3]:


if __name__=='__main__':
    main()

