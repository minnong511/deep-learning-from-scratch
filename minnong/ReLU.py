class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # ReLU는 mask라는 instance 변수를 가짐.
        # mask는 True/False 로 구성된 numpy 배열
        self.mask = (x<=0)
        # x가 0이하라면 True, 반대의 경우는 False
        # 근데
        out = x.copy
        out[self.mask] = 0

        return out

    def backward(self,dout):
        dout[self.mask] = 0
        # True 면 ReLU 함수가 비활성된 것이므로, 0을 전파함.
        dx = dout

        return dx