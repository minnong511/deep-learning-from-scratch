import numpy as np

class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        # 훈련 시에는 순전파때마다 self.mask에 삭제할 뉴런을 False로 표시함.
        # self.mask는 x와 형상이 같은 배열을 무작위로 생성
        # 그 값이 dropout_ratio보다 큰 원소만 True로 설정함.
        # 역전파의 동작은 ReLU 와 동일함.

    def forward(self, X, train_flag =True):
        if train_flag:
            self.mask = np.random.rand(*X.shape) > self.dropout_ratio
            # 기준치 이상만 값을 냄겨놓은다.
            return X * self.mask
        else:
            return X * (1.0 - self.dropout_ratio) # 근데 굳이 비율을 더 곱해줄 필요가 있나..?

    def backward(self, dout):
        return dout * self.mask