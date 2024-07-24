#Affine

from common.layers import *
from collections import OrderedDict


class TwoLayerNet:

    # 초기화를 수행함.
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 가중치 초기화
        # params => 신경망의 매개변수를 보관
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        # 신경망의 계층을 보관
        self.layers = OrderedDict()
        #Orderedict는 순서가 있는 dictionary를 말하는 것임. -> 딕셔너리에 추가한 순서를 기억함.
        #순전파 때는 추가한 순서대로 각 계층의 forward() 메서드를 호출하기만 하면 됨.
        #마찬가지로 Backpropagation 때는 반대로만 호출하면 되므로 매우 편리함.
        #이런 Trick을 잘 알고 있는 것이 깔끔한 구현으로 이어진다고 생각함.
        self.layers["Affine1"] = Affine(self.params["W1"],self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"],self.params["b2"])
        #lastlayer => 신경망의 마지막 계층.
        self.lastLayer = SoftmaxWithLoss()

        #예측 추론 , x는 image data
        def predict(self, x):
            for layer in self.layers.values():
                # values 를 사용하면 딕셔너리에 내에 있는 key : value 중 Value를 뽑아온다.
                x = layer.forward(x)
                # 신경망이 앞으로 계속 들어가면서 순환하게 된다.
            return x

        def loss(self, x, t):
            y = self.predict(x)
            return self.lastLayer.loss(y, t)

        def accuracy(self, x, t):
            y = self.predict(x)
            # 데이터의 형태를 잘 생각해보면 axis = 1 이어야 함.  -> 왜냐하면 Y = (N,정답 행렬) 이렇게 저장되기 때문에.
            # axis = 1 각 행을 따라 가장 큰 값을 찾는다.
            y = np.argmax(y, axis=1)
            if t.ndim != 1 : t = np.argmax(t, axis=1)

            accuracy = np.sum(y==t) /  float(x.shape[0])
            return accuracy

        #  수치미분으로 구하는 gradient
        def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x, t)

            # 결과를 저장한다.
            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
            return grads

        # backpropagation으로 구하는 기울기값.
        def gradient(self, x, t):

            self.loss(x,t)
            # loss function 안에 predict 함수가 내장되어 있으므로, 자동으로 forward pass를 하게 된다.

            dout = 1
            dout = self.lastLayer.backward(dout)

            layers = list(self.layers.values())
            layers.reverse()
            # 딕셔너리를 뒤집어서 순차적으로 계산하게 된다. -> 오차역전파 할 때는 반대인거 기억하제?

            for layer in layers:
                dout = layer.backward(dout)
                # 너는 어디서 나온거니??초

            # 결과를 저장한다.
            grads = {}
            grads["W1"] = self.layers["Affine1"].dW
            grads["b1"] = self.layers["Affine1"].db
            grads["W2"] = self.layers["Affine2"].dW
            grads["b2"] = self.layers["Affine2"].db
            return grads

        # 구조가 직관적이지는 않은 거 같다..
