import numpy as np


class AdaGrad:

    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h  = None

    # 수식을 그냥 코드로 나타낸 것일 뿐...
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            # 딕셔너리에 값을 받아옴.
            for key ,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)