import sys, os
sys.path.append(os.pardir)
from Activation_function import *
from mnist import load_mnist
from gradient import numerical_gradient # 편미분 기울기 구하기

##############################데이터 부르기#########
(x_train,y_train),(x_test,y_test) = load_mnist(
    normalize=False,flatten=True)
###################################################

class TwoLayerNet:
    def __init__(self):
        self.params={}
        self.params['W1'] = np.random.rand(784,392)
        self.params['W2'] = np.random.rand(392,10)

    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        a1 = np.dot(x,W1)
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)
        z2= sigmoid(a2)
        y = softmax(z2)
        return y

    # x: 입력데이터, t: 정답 레이블
    def loss (self,x,t):
        y= self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y =self.predict(x)
        y= np.argmax(y,axis=1)
        t= np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads={}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])

        return grads

