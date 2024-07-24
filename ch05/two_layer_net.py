# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        # 클래스를 불러오고, 불러온 클래스 내에 parameter들을 저장하게 된다. 저장된 parameter에는 행렬이 들어가게 되어있다.
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        # 여기서 layer는 Affine을 받아왔으므로 Affine 내에 있는 forward 함수를 사용한다. -> 행렬 곱으로 연산이 된다.
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        # 사실이 실행이 되는 부분은 이 부분이 아닐까...?
        y = self.predict(x)
        # lastlayer 는 SoftmaxWithLoss 로 받아 왔으르모 따로 계산한다.
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        # 여기서 loss를 실행시키면 바로 predict 까지 이어짐
        # 근데 numerical gradient를 위한 loss function은 어떻게 되는거지?

        loss_W = lambda W: self.loss(x, t)
        # loss_W 가 numerical gradient를 위한 함수임.
        # 헷갈리면 page 135 ~ 136 을 참고할 수 있도록.
        # 이걸로 loss 함수를 불러온다.
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W    , self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        # 아마 행렬로 저장되어 있겠지.
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        # 클래스 내에 잘 저장되어 있으므로, 불러오면 된다.
        # 위에서 Affine1 과 Affine2 잘 분리해놓았으니..
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
