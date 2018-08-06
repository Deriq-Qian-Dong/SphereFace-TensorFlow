import tensorflow as tf
import numpy as np
import cv2
import os
from matlab_cp2tform import get_similarity_transform_for_cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as plt1
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
import math
from random import shuffle
losslist=[]
acclist=[]
epochlist=[]
def Loss_ASoftmax(x, y, l, num_cls, m = 2, name = 'asoftmax'):
    '''
    x: B x D - data
    y: B x 1 - label
    l: 1 - lambda
    '''
    xs = x.get_shape()
    w = tf.get_variable("asoftmax/W", [xs[1], num_cls], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

    eps = 1e-8

    xw = tf.matmul(x,w)

    if m == 0:
        return xw, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=xw))

    w_norm = tf.norm(w, axis = 0) + eps
    logits = xw/w_norm

    if y is None:
        return logits, None

    ordinal = tf.constant(list(range(0, xs[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, y], axis = 1)

    x_norm = tf.norm(x, axis = 1) + eps

    sel_logits = tf.gather_nd(logits, ordinal_y)

    cos_th = tf.div(sel_logits, x_norm)

    if m == 1:

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                              (labels=y,  logits=logits))

    else:

        if m == 2:

            cos_sign = tf.sign(cos_th)
            res = 2*tf.multiply(cos_sign, tf.square(cos_th)) - 1

        elif m == 4:

            cos_th2 = tf.square(cos_th)
            cos_th4 = tf.pow(cos_th, 4)
            sign0 = tf.sign(cos_th)
            sign3 = tf.multiply(tf.sign(2*cos_th2 - 1), sign0)
            sign4 = 2*sign0 + sign3 - 3
            res = sign3*(8*cos_th4 - 8*cos_th2 + 1) + sign4
        else:
            raise ValueError('unsupported value of m')

        scaled_logits = tf.multiply(res, x_norm)

        f = 1.0/(1.0+l)
        ff = 1.0 - f
        comb_logits_diff = tf.add(logits, tf.scatter_nd(ordinal_y, tf.subtract(scaled_logits, sel_logits), logits.get_shape()))
        updated_logits = ff*logits + f*comb_logits_diff

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits))
    return logits, loss
def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos+neg

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

datapath='/home/scut/dq/PycharmProjects/datasets/lfw'
flist=[]
for(dirpath,dirnames,filenames) in os.walk(datapath):
    for filename in filenames:
        if 'CASIA-WebFace' in dirpath:continue
        flist.append(dirpath+'/'+filename)
label=[]
for f in flist:
    label.append(f.split('/')[-2])
classnum=len(set(label))
labelset = list(set(label))
for i in range(len(label)):
    label[i] = labelset.index(label[i])
# sess = tf.Session()
# label = sess.run(tf.one_hot(label, classnum))
label=np.array(label)
label=label.reshape(-1,1)
flist = np.array(flist)
flist = flist.reshape(-1, 1)
idx=[i for i in range(len(flist))]
shuffle(idx)

def get_batch(step=0,bz=64):
    x=flist[idx[step*bz:(step+1)*bz]]
    y=label[idx[step*bz:(step+1)*bz]]
    resultx=[]
    for ix in x:
        img=plt.imread(ix[0])
        tmp = detector.detect_faces(img)[0]['keypoints']
        landmark = []
        for t in tmp.items():
            for tt in t[1]:
                landmark.append(tt)
        img=alignment(img,landmark)
        resultx.append(img)
    resultx=np.array(resultx)
    y=y.reshape(bz,)
    return resultx,y


class SphereFace():
    def __init__(self,bz,m=4):
        self.class_num=classnum
        self.m=m
        self.lr=0.01
        self.batch=bz
        self.x=tf.placeholder(dtype=tf.float32,shape=[self.batch,112,96,3])#[b,112,96,3]
        self.y=tf.placeholder(dtype=tf.int64,shape=[self.batch])
        self.conv1_1=self.add_conv(filter_shape=[3,3,3,64],
                                 bias_shape=[64],
                                 input=self.x,strides=2,)#[b,56,48,64]
        # self.test=self.conv1_1#[64, 56, 48, 64]
        with tf.variable_scope('relu1_1'):
            self.relu1_1=prelu(self.conv1_1)
        # self.test=self.relu1_1
        self.result=self.relu1_1
        self.conv1_2=self.add_conv(filter_shape=[3,3,64,64],input=self.result,
                                   bias_shape=[64])
        # self.test=self.conv1_2
        with tf.variable_scope('relu1_2'):
            self.relu1_2=prelu(self.conv1_2)
        self.conv1_3=self.add_conv(filter_shape=[3,3,64,64],input=self.relu1_2,
                                   bias_shape=[64])
        with tf.variable_scope('relu1_3'):
            self.relu1_3=prelu(self.conv1_3)
        self.result+=self.relu1_3

        self.conv2_1=self.add_conv(filter_shape=[3,3,64,128]
                                   ,bias_shape=[128],input=self.result,strides=2)#[b,28,24,128]
        with tf.variable_scope('relu2_1'):
            self.relu2_1=prelu(self.conv2_1)
        self.result = self.relu2_1
        self.conv2_2=self.add_conv(filter_shape=[3,3,128,128],bias_shape=[128],input=self.relu2_1)
        with tf.variable_scope('relu2_2'):
            self.relu2_2=prelu(self.conv2_2)
        self.conv2_3 = self.add_conv(filter_shape=[3,3,128,128],bias_shape=[128],input=self.relu2_2)
        with tf.variable_scope('relu2_3'):
            self.relu2_3 = prelu(self.conv2_3)
        self.result+=self.relu2_3

        self.conv2_4 = self.add_conv(filter_shape=[3,3,128,128],bias_shape=[128],input=self.result)
        with tf.variable_scope('relu2_4'):
            self.relu2_4 = prelu(self.conv2_4)
        self.conv2_5 = self.add_conv(filter_shape=[3,3,128,128],bias_shape=[128],input=self.relu2_4)
        with tf.variable_scope('relu2_5'):
            self.relu2_5 = prelu(self.conv2_5)
        self.result+=self.relu2_5
        # self.test=self.relu2_5


        self.conv3_1 = self.add_conv(filter_shape=[3,3,128,256],bias_shape=[256]
                                     ,input=self.result,strides=2) #[b,14,12,256]
        with tf.variable_scope('relu3_1'):
            self.relu3_1 = prelu(self.conv3_1)
        self.result = self.relu3_1
        self.conv3_2 = self.add_conv(filter_shape=[3,3,256,256],bias_shape=[256],input=self.relu3_1)
        with tf.variable_scope('relu3_2'):
            self.relu3_2 = prelu(self.conv3_2)
        self.conv3_3 = self.add_conv([3,3,256,256],[256],self.relu3_2)
        with tf.variable_scope('relu3_3'):
            self.relu3_3 = prelu(self.conv3_3)
        self.result+=self.relu3_3

        self.conv3_4 = self.add_conv([3,3,256,256],[256],self.result)
        with tf.variable_scope('relu3_4'):
            self.relu3_4 = prelu(self.conv3_4)
        self.conv3_5 = self.add_conv([3,3,256,256],[256],self.relu3_4)
        with tf.variable_scope('relu3_5'):
            self.relu3_5 = prelu(self.conv3_5)
        self.result+=self.relu3_5

        self.conv3_6 = self.add_conv([3,3,256,256],[256],self.result)
        with tf.variable_scope('relu3_6'):
            self.relu3_6 = prelu(self.conv3_6)
        self.conv3_7 = self.add_conv([3,3,256,256],[256],self.relu3_6)
        with tf.variable_scope('relu3_7'):
            self.relu3_7 = prelu(self.conv3_7)
        self.result+=self.relu3_7

        self.conv3_8 = self.add_conv([3,3,256,256],[256],self.result)
        with tf.variable_scope('relu3_8'):
            self.relu3_8 = prelu(self.conv3_8)
        self.conv3_9 = self.add_conv([3,3,256,256],[256],self.relu3_8)
        with tf.variable_scope('relu3_9'):
            self.relu3_9 =prelu(self.conv3_9)
        # self.test=self.relu3_9
        self.result+=self.relu3_9

        self.conv4_1 = self.add_conv([3,3,256,512],[512],self.result,strides=2) #[b,7,6,512]
        with tf.variable_scope('relu4_1'):
            self.relu4_1 =prelu(self.conv4_1)
        self.result = self.relu4_1
        self.conv4_2 = self.add_conv([3,3,512,512],[512],self.relu4_1)
        with tf.variable_scope('relu4_2'):
            self.relu4_2 =prelu(self.conv4_2)
        self.conv4_3 = self.add_conv([3,3,512,512],[512],self.relu4_2)
        with tf.variable_scope('relu4_3'):
            self.relu4_3 = prelu(self.conv4_3)
        self.result+=self.relu4_3

        self.poutput=self.add_dense(self.result)
        # if(self.m==1):
        #     self.output=tf.nn.softmax(self.poutput)
        #     self.loss=tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.output+1e-10)))
        # else:
        self.output,self.loss=Loss_ASoftmax(self.poutput,self.y,l=1.0,num_cls=self.class_num,m=self.m)
        self.train_step=tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        corrects=tf.equal(tf.argmax(self.output,1),self.y)
        self.acc=tf.reduce_mean(tf.cast(corrects,tf.float32))

        self.test=tf.argmax(self.output,1)

        self.init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(self.init)
    def add_dense(self,input):
        input=tf.reshape(input,[-1,7*6*512])
        output=tf.layers.dense(inputs=input,units=512)
        return output

    def weight_variable(self,shape):
        init=tf.truncated_normal(shape=shape,stddev=0.01)
        return tf.Variable(init)

    def add_conv(self,filter_shape,bias_shape,input,strides=1,padding='SAME'):
        w1=self.weight_variable(filter_shape)
        b1=self.weight_variable(bias_shape)
        return tf.nn.conv2d(input,w1,strides=[1,strides,strides,1],padding=padding)+b1
    
from matplotlib.patches import Rectangle
if __name__=='__main__':
    bz=64
    model=SphereFace(bz)

    # x, y = get_batch(step=0, bz=bz)
    # test=model.sess.run(model.test , feed_dict={model.x: x, model.y: y})
    # print(test)
    # print(test.shape)

    for epoch in range(20):
        epochlist.append(epoch)
        print('******epoch%d*******'%(epoch+1))
        for i in range(int(len(flist)/bz)):
            x, y = get_batch(step=0,bz=bz)
            #print(y)
            model.sess.run(model.train_step,feed_dict={model.x: x, model.y: y})
            # y = model.sess.run(model.y,feed_dict={model.x: x, model.y: y})
            # t = model.sess.run(model.output, feed_dict={model.x: x, model.y: y})
            # print(y.shape, t.shape)
            loss = model.sess.run(model.loss, feed_dict={model.x: x, model.y: y})
            acc = model.sess.run(model.acc ,feed_dict={model.x: x, model.y: y})
            test=model.sess.run(model.test , feed_dict={model.x: x, model.y: y})
            #print(test)
            print('step:%d / %d  loss:%f  acc:%f'%(i+1,int(len(flist)/bz),loss,acc))
        losslist.append(loss)
        acclist.append(acc)
    np.save('a_loss.npy',losslist)
    np.save('a_acc.npy', acclist)
    # plt1.xlabel('epoch')
    # plt1.ylabel('loss&acc')
    # plt_dict={}
    # plt_dict['loss']=losslist
    # plt_dict['acc']=acclist
    # plt_color_array=['blue','gree']
    # proxy=[]
    # legend_array=[]
    # for index,(tmp,lora)in enumerate(plt_dict.items()):
    #     color=plt_color_array[index]
    #     plt1.plot(range(len(epochlist)),epochlist,'-%s'%color[0])
    #     proxy.append(Rectangle((0,0),0,0,facecolor=color))
    #     legend_array.append(tmp)
    # plt1.legend(proxy,legend_array)
    # plt1.show()
    saver=tf.train.Saver()
    saver.save(model.sess,'finalmodel.ckpt')

#################about 20 epochs to meet a good result#########################
