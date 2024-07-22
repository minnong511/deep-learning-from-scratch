import numpy as np
from dataset.mnist import load_mnist
from ch04.two_layer_net import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

iters_num = 10000 # 반복횟수
train_size = x_train.shape[0] # 60000개
batch_size = 100 # 샘플로 뽑아낼 데이터 갯수
learning_rate = 0.1 # 학습률

# Layer 갯수 초기화  :  input = image(width * depth) , hidden = 은닉층 , output size
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 미니배치 획득
    # 무작위로 100개를 뽑는다.
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# backpropagation 을 알고 강의를 들어서 그런가.. 더 헷갈리네...