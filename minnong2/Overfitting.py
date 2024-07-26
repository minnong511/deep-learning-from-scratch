from dataset.mnist import load_mnist
from common.layers import *

(x_train , t_train) , (x_test,t_test) = load_mnist(normalize=True) # 데이터셋을 불러온다

# 필요한 학습 데이터 수를 정한다
x_train = x_train[:300]
t_train = t_train[:300]
# 여기서 필요한 데이터 수를 300개로 정했다. -> batch 사이즈와 헷갈리지 마라!

network = MultilayerNet(input_size=784, hidden_size=[100,100,100,100,100], output_size=10) # 굳이 클래스를 불러오지는 않을게... , 학습시킬 모델을 불러온다
optimizer = Adam(lr=0.0001) # 매개변수 갱신방식 고르기
max_epochs = 200 # 최대 반복 숫자
train_size = x_train.shape[0] # train 데이터 수를 뽑아온다.
batch_size = 100 # 배치 사이즈, 훈련시에 사용할 데이터의 수를 골라온다

train_loss_list = []
test_acc_list = []
train_acc_list = [] # 검출 결과를 검출할 데이터 공간을 만들어준다.

iter_per_epoch = train_size // batch_size # 데이터셋 한 번을 학습하는데 필요한 반복 횟수.
epoch_cnt = 0

for i in range(10000000):
    batch_mask = np.random.choice(train_size, batch_size) # train data 중에 batch 갯수 만큼 무작위로 뽑는다.
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(grads, network.parameters())

    if i % iter_per_epoch == 0:
        # 특정횟수를 지나갈 때마다 정확도를 리스트에 추가한다.
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 최소 반복수를 맞추면 epoch를 돌린 것이기 떄문에, cnt를 한 번 추가해 준다.
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

