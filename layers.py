import numpy as np
import math


class Linear():
    # DO NOT DELETE
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b.reshape(1, -1)
        return self.out

    def backward(self, delta):
        self.db = delta.sum(axis=0)
        self.dW = np.dot(delta.T, self.x)
        dx = np.dot(delta, self.W)
        return dx

        

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        #num channels in input
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.width = None
        self.batch = None
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        ## Your codes here
        # out = bias(out_channel) + sum over in_channel(weight[out_channel][kernel_size] * input(batch, kernel_size)
        self.x = x
        self.batch, in_channel, self.width = x.shape
        assert in_channel == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)

        out_width = math.floor((self.width - self.kernel_size) / self.stride)+1

        toReturn = np.zeros((self.batch, self.out_channel, out_width))

        for idx, i in enumerate(range(0, self.width - self.kernel_size + 1, self.stride)):
            slice = x[:,:,i:i+self.kernel_size].reshape(self.batch, -1)
            filter = self.W.reshape(self.out_channel, -1)
            z = slice.dot(filter.T) + self.b
            toReturn[:,:,idx] = z

        return toReturn

    def backward(self, delta):
        
        ## Your codes here
        self.db = np.sum(delta, axis=(0,2))
        self.dW = np.zeros_like(self.W)
        dx = np.zeros_like(self.x)

        # loop over third dim of delta, the out width
        for i in range(delta.shape[2]):
            # slice before reshape is 2,10,3 aka batch,in_channel, kernel_size
            slice = self.x[:, :, i*self.stride:i*self.stride+self.kernel_size].reshape(self.batch, -1)
            # whole delta is 2,7,9 aka batch, out_channel, out_width
            # relevant delta is 2,7
            relevant_delta = delta[:,:,i]
            # dW is 7,10,3 because W is 7,10,3 aka out_channel, in_channel, kernel_size
            this_dW = slice.T.dot(relevant_delta)
            self.dW += this_dW.T.reshape(self.out_channel, self.in_channel, self.kernel_size)
            this_dx = relevant_delta.dot(self.W.reshape(self.out_channel, -1)).reshape(self.batch, self.in_channel, -1)
            dx[:, :, i*self.stride:i*self.stride+self.kernel_size] += this_dx
        return dx


# rint = np.random.randint
# norm = np.linalg.norm
# in_c, out_c = 10, 7
# kernel, stride = 3, 2
# batch, width = 2, 20
# net = Conv1D(in_c, out_c, kernel, stride)
# x = np.random.randn(batch, in_c, width)
# y = net(x)
# b,c,w = y.shape
# delta = np.random.randn(b, c, w)
# net.backward(delta)

class Flatten():
    def __init__(self):
        self.shape = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta):
        return delta.reshape(self.shape)


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x>=0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta