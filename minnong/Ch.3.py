# 내부 라이브러리를 불러오는 구간
import os
import pickle
import sys
import numpy as np

# 데이터셋을 불러오는 구간
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

# 데이터셋 저장


def relu(x):
    return np.maximum(0,x)
def get_data():
    (x_train , t_train ), (x_test , t_test) = load_mnist(flatten=True , normalize= False)
    return x_test,t_test

# 정답 레이블이 one-hot-encoding 이 되어있는 경우는 이렇게한다.
def cross_entropy_error(y,t):
    if y.ndim == 1:
        # reshape 함수로 형상을 어떻게 바꿀까?
        # 데이터의 일괄 처리를 위해 행렬로 식을 변경.
        # Batch size에 맞춰서 배열을 변환.
        # t가 one-hot-encoding 이 되어있어야 한다.
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7) / batch_size)

def cross_entropy_error_with_no_encoding(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]) / batch_size)




def init_network():
    # sample_weight_pkl에는 저장되어 있는 "학습된 가중치 매개변수"를 읽음.
    # weight 와 bias가 dictionary 변수로 저장되어 있음.
    # 동일 directory안에 파일을 넣던가 혹은 경로를 따로 지정해줘야 함.
    with open("sample_weight.pkl" , "rb") as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1 # 행렬 연산
    z1 = relu(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, W3) + b3
    y = relu(a3)

    return y

x,t = get_data()
network = init_network()
batch_size = 100
# 100개의 이미지를 한 번에 연산한다.
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    # x[0:100], x[100,200], x[200:300], ....... 이렇게 증가하게 된다.
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    # 최댓값의 인덱스를 가져온다. -> 0차원 , 1차원, 2차원 순으로 가져오게 되어있다.
    # 만약 100 * 10 의 행렬이라면 100(행)이 0차원이고 , 10이 1차원임. ㅇㅇ
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    # argmax로 뽑아낸 값과 기존 정답간의 데이터를 비교새함.

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))



#===========================================================================#
# normalize : 입력 이미지의 픽셀을 0.0 ~ 1.0 사이로 정규화 (normalize = True)
# flatten : 입력 이미지를 1차원 배열로 만들지 정함 (flatten = True )
# one_hot_label : one-hot 방식 (0과 1로만 저장)
#===========================================================================#

#img = x_train[0]  # x_train -> (60000 , 784) , 60000장의 학습 이미지 28*28 크기의 이미지
#label = t_train[0] # t_train -> (60000) 60000개의 레이블 -> 사진의 정답이 들어있다.


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    # fromarray로 이미지 데이터 객체로 변환.
    pil_img.show()


def relu(x):
    return np.maximum(0,x)
