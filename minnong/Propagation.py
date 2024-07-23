class Mullayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class Addlayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out

    def backward(self,dout):
        dx = dout
        dy = dout
        return dx, dy

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) /batch_size

        return dx



