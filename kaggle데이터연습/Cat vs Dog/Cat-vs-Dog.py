"""라이브러리, 데이터 불러오기"""
import tensorflow as tf
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from load_image import load_train_image
from load_image import load_test_image
""""""""""""""""""
train_image,train_label = load_train_image()
test_image,test_label = load_test_image()

'''1이 없어서 만들어주기'''
train_label[12500:]=1
train_label= train_label.reshape(25000,-1)

'''shuffle: train_data'''
train_num = np.random.choice(train_label.shape[0],25000)
train_image = train_image[train_num]
train_label = train_label[train_num]
#np.random.shuffle()
tf.reset_default_graph()

'''Input placeholder'''
X = tf.placeholder(tf.float32,[None,64,64,3])
x_img = tf.reshape(X,[-1,64,64,3]) # img 64*64*3, -1: N개
Y = tf.placeholder(tf.float32,[None,1])
#keep_prob = tf.placeholder(tf.float32)

'''Convolution layer 1 '''
#L1 input shape (?,64,64,3)
W1 = tf.Variable(tf.random_normal([3,3,3,32],stddev=0.01))
#Conv -> (?,64,64,32)
#Pool -> (?,32,32,32)
L1 = tf.nn.conv2d(x_img,W1,strides=[1,1,1,1],
                  padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1,ksize=(1,2,2,1),strides=[1,2,2,1],
                    padding='SAME')

L1.shape
'''Convolution Layer 2 '''
#L2 input shape(32,32,32)
W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
#Conv ->(?,32,32,64)
#Pool ->(?,16,16,64)
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],
                  padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME')

''' Convolution Layer 3 '''
#L3 input shape(16,16,64)
W3 = tf.Variable(tf.random_normal([3,3,64,32],stddev=0.01))
#Conv ->(?,16,16,64)
#Pool ->(?,8,8,64)
L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],
                  padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME')

''' Convolutuin layer 4 '''
#L3 input shape(8,8,32)
W4 = tf.Variable(tf.random_normal([8,8,32,32],stddev=0.01))
L4 = tf.nn.conv2d(L3,W4,strides=[1,1,1,1],
                  padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4,ksize=[1,2,2,1],strides=[1,2,2,1],
                    padding='SAME') 
#L4.shape = (None,4,4,32)

'''Fully connected layer '''
L4=tf.reshape(L4,[-1,4*4*32])

#Final FC 8*8*64 inputs -> 1 outputs
W5 = tf.get_variable('W5',shape=[4*4*32,1],
                     initializer=tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(L4,W5)+b)
#hypothesis = tf.nn.dropout(hypothesis, keep_prob= keep_prob)

#define cost/loss & optimizer
#cost = -tf.reduce_mean(Y*tf.log(hypothesis)+ (1-Y)* tf.log(1-hypothesis))

cost = -tf.reduce_mean(Y*tf.log(hypothesis+0.01)+ (1-Y)*
                      tf.log(1-hypothesis+0.01))


#optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005,momentum=0.9).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


''' Accuracy computation '''
predicted = tf.cast(hypothesis >0.5,dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),
                                  dtype=tf.float32))



#initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_size=train_image.shape[0]
traing_epochs = 20
batch_size  = 250

#train my model
print('Learning started. It takes some time.')

for epoch in range(traing_epochs):
  avg_cost = 0
  avg_acc = 0
  for i in range(100):
    batch_mask = np.random.choice(train_size,batch_size)
    batch_xs, batch_ys = train_image[batch_mask],train_label[batch_mask]
    c,_,a = sess.run([cost,optimizer,accuracy],
                    feed_dict = {X: batch_xs, Y: batch_ys})
    avg_cost += c /100
    avg_acc += a /100
  print('Epoch:','%04d' % (epoch+1),'cost=','{:.9f}'.format(avg_cost),'accuracy=','{:.9f}'.format(avg_acc))
#batch_mask reset
#batch_mask = np.random.choice(train_size,batch_size)
#h,c,a = sess.run([hypothesis,predicted,accuracy],
#                 feed_dict={X:train_image[batch_mask],Y:train_label[batch_mask]})
#print('Train Accuracy:',a)

print('Learning Finished!')

#Test model and check accuracy22
kk=[]
for i in range(test_image.shape[0]):
    test_y = sess.run(hypothesis,feed_dict={X:test_image[i:i+1]})
    if test_y>= 0.5:
        kk.append(1)
    else:
        kk.append(0)
        
'''submit answer'''
submission = pd.read_csv('C:/Users/ehdrb/Desktop/데이터/catdog/sampleSubmission.csv')

num=0
for i in kk:
    submission['label'][num]=i
    num+=1

submission.to_csv('donggyu_sub.csv',index=False)

'''
Learning started. It takes some time.
Epoch: 0001 cost= 0.743522353 accuracy= 0.503360001
Epoch: 0002 cost= 0.673226033 accuracy= 0.503840003
Epoch: 0003 cost= 0.671894540 accuracy= 0.515000001
Epoch: 0004 cost= 0.664129316 accuracy= 0.546920000
Epoch: 0005 cost= 0.621692169 accuracy= 0.626719999
Epoch: 0006 cost= 0.570518679 accuracy= 0.686960003
Epoch: 0007 cost= 0.535969126 accuracy= 0.717720000
Epoch: 0008 cost= 0.471001709 accuracy= 0.762359998
Epoch: 0009 cost= 0.447456948 accuracy= 0.780080000
Epoch: 0010 cost= 0.419660189 accuracy= 0.798559999
Epoch: 0011 cost= 0.363559155 accuracy= 0.835599999
Epoch: 0012 cost= 0.349796972 accuracy= 0.842600001
Epoch: 0013 cost= 0.291813820 accuracy= 0.868199999
Epoch: 0014 cost= 0.269606369 accuracy= 0.883440000
Epoch: 0015 cost= 0.238324076 accuracy= 0.899160000
Epoch: 0016 cost= 0.295278777 accuracy= 0.871079999
Epoch: 0017 cost= 0.210882574 accuracy= 0.913999999
Epoch: 0018 cost= 0.220399855 accuracy= 0.906840001
Epoch: 0019 cost= 0.236315680 accuracy= 0.902919999
Epoch: 0020 cost= 0.212344801 accuracy= 0.916119999
Learning Finished!
'''


