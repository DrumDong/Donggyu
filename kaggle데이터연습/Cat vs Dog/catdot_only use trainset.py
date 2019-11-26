import tensorflow as tf
import numpy as np
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from load_image import load_train_image
#from load_image import load_test_image
""""""""""""""""""
real_train_image,real_train_label = load_train_image()
#test_image,test_label = load_test_image()

real_train_label[12500:]=1
real_train_label= real_train_label.reshape(25000,-1)

'''data split'''
test_num = np.random.choice(real_train_label.shape[0],1000)

train_num=[]
for i in range(0,25000):
    if i not in test_num:
        train_num.append(i)
train_num = np.array(train_num)

train_image = real_train_image[train_num]
train_label = real_train_label[train_num]
test_image = real_train_image[test_num]
test_label = real_train_label[test_num]

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


optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005,momentum=0.9).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(cost)
#optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

''' Accuracy computation '''
predicted = tf.cast(hypothesis >0.5,dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),
                                  dtype=tf.float32))



#initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_size=train_image.shape[0]
traing_epochs = 30
batch_size  = 240

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
print('Test Accuracy',sess.run(accuracy,
                               feed_dict={X:test_image,Y:test_label}))

''' 1.
cost = -tf.reduce_mean(Y*tf.log(hypothesis+0.01)+ (1-Y)*
                      tf.log(1-hypothesis+0.01))
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(cost)
Learning started. It takes some time.
Epoch: 0001 cost= 0.663826604 accuracy= 0.553666664
Epoch: 0002 cost= 0.641853099 accuracy= 0.591874997
Epoch: 0003 cost= 0.562527082 accuracy= 0.694041668
Epoch: 0004 cost= 0.497710277 accuracy= 0.747083329
Epoch: 0005 cost= 0.448941556 accuracy= 0.778125001
Epoch: 0006 cost= 0.407072958 accuracy= 0.803041668
Epoch: 0007 cost= 0.355432712 accuracy= 0.832958333
Epoch: 0008 cost= 0.334746358 accuracy= 0.842541664
Epoch: 0009 cost= 0.280532609 accuracy= 0.872999998
Epoch: 0010 cost= 0.269773770 accuracy= 0.880166666
Epoch: 0011 cost= 0.244364778 accuracy= 0.894208333
Epoch: 0012 cost= 0.212324655 accuracy= 0.909250001
Epoch: 0013 cost= 0.221207934 accuracy= 0.903999999
Epoch: 0014 cost= 0.208536242 accuracy= 0.910125000
Epoch: 0015 cost= 0.183909689 accuracy= 0.919250004
Learning Finished!
Test Accuracy 0.778
'''

'''
Learning started. It takes some time.
Epoch: 0001 cost= 0.674938259 accuracy= 0.540208333
Epoch: 0002 cost= 0.630776679 accuracy= 0.617874999
Epoch: 0003 cost= 0.599926006 accuracy= 0.651666667
Epoch: 0004 cost= 0.549416901 accuracy= 0.705541666
Epoch: 0005 cost= 0.492417949 accuracy= 0.751250001
Epoch: 0006 cost= 0.447776116 accuracy= 0.778500000
Epoch: 0007 cost= 0.419199760 accuracy= 0.797166666
Epoch: 0008 cost= 0.387132298 accuracy= 0.815749999
Epoch: 0009 cost= 0.357750674 accuracy= 0.833833330
Epoch: 0010 cost= 0.315204444 accuracy= 0.853916669
Epoch: 0011 cost= 0.285071390 accuracy= 0.870458333
Epoch: 0012 cost= 0.271111649 accuracy= 0.881083333
Epoch: 0013 cost= 0.249229645 accuracy= 0.890874997
Epoch: 0014 cost= 0.223676317 accuracy= 0.901791667
Epoch: 0015 cost= 0.209725726 accuracy= 0.908875005
Epoch: 0016 cost= 0.208703479 accuracy= 0.909416667
Epoch: 0017 cost= 0.199338793 accuracy= 0.917041665
Epoch: 0018 cost= 0.184614787 accuracy= 0.920583335
Epoch: 0019 cost= 0.174246031 accuracy= 0.927791664
Epoch: 0020 cost= 0.172192289 accuracy= 0.928625001
Learning Finished!
Test Accuracy 0.789
'''

'''
feed_dict={X:test_image,Y:test_label}))
Learning started. It takes some time.
Epoch: 0001 cost= 0.676308712 accuracy= 0.545166665
Epoch: 0002 cost= 0.615842771 accuracy= 0.633166664
Epoch: 0003 cost= 0.554844259 accuracy= 0.701375002
Epoch: 0004 cost= 0.504446954 accuracy= 0.739416664
Epoch: 0005 cost= 0.450837769 accuracy= 0.779833332
Epoch: 0006 cost= 0.406588172 accuracy= 0.801583333
Epoch: 0007 cost= 0.360592747 accuracy= 0.829666663
Epoch: 0008 cost= 0.323035327 accuracy= 0.852375002
Epoch: 0009 cost= 0.309598818 accuracy= 0.858958333
Epoch: 0010 cost= 0.293507924 accuracy= 0.868166667
Epoch: 0011 cost= 0.271800652 accuracy= 0.879916666
Epoch: 0012 cost= 0.222767009 accuracy= 0.904166667
Epoch: 0013 cost= 0.227692541 accuracy= 0.903541667
Epoch: 0014 cost= 0.195610862 accuracy= 0.917916669
Epoch: 0015 cost= 0.215328405 accuracy= 0.908791668
Epoch: 0016 cost= 0.174096441 accuracy= 0.928000001
Epoch: 0017 cost= 0.190271339 accuracy= 0.919541671
Epoch: 0018 cost= 0.183866934 accuracy= 0.923125001
Epoch: 0019 cost= 0.201936713 accuracy= 0.917916668
Epoch: 0020 cost= 0.167800497 accuracy= 0.930208333
Epoch: 0021 cost= 0.231834566 accuracy= 0.908375000
Epoch: 0022 cost= 0.178773995 accuracy= 0.928791667
Epoch: 0023 cost= 0.221981051 accuracy= 0.913166668
Epoch: 0024 cost= 0.200310150 accuracy= 0.921458336
Epoch: 0025 cost= 0.187634619 accuracy= 0.928458333
Epoch: 0026 cost= 0.210400553 accuracy= 0.920541670
Epoch: 0027 cost= 0.283560806 accuracy= 0.898083334
Epoch: 0028 cost= 0.247323482 accuracy= 0.910458336
Epoch: 0029 cost= 0.358437587 accuracy= 0.875833331
Epoch: 0030 cost= 0.271284738 accuracy= 0.896708335
Learning Finished!
Test Accuracy 0.756
'''
