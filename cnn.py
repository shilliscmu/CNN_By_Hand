from layers import *

class CNN_B():
    def __init__(self):
        # Your initialization code goes here
        self.layers = []
        # in_channel, out_channel, kernel_size, stride
        self.layers.append(Conv1D(24, 8, 8, 4))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(8, 16, 1, 1))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(16, 4, 1, 1))

        self.layers.append(Flatten())

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # W is out_channel, in_channel, kernel_size
        # weights is layer, input_width*kernel_size, out_channels

        # weights[0] = np.transpose(weights[0])
        # weights[0] = np.reshape(weights[0], (8, 8, 24))
        # weights[0] = np.transpose(weights[0], (0, 2, 1))

        self.layers[0].W = np.transpose(np.reshape(weights[0], (8, 24, 8)), (2, 1, 0))
        self.layers[2].W = np.transpose(np.reshape(weights[1], (1, 8, 16)), (2, 1, 0))
        self.layers[4].W = np.transpose(np.reshape(weights[2], (1, 16, 4)), (2, 1, 0))

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

class CNN_C():
    def __init__(self):
        # Your initialization code goes here
        # in_channel, out_channel, kernel_size, stride

        self.layers = []
        self.layers.append(Conv1D(24, 2, 2, 2))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(2, 8, 2, 2))
        self.layers.append(ReLU())
        self.layers.append(Conv1D(8, 4, 2, 1))

        self.layers.append(Flatten())


    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        self.layers[0].W = np.transpose(np.reshape(weights[0][:48, :2], (2, 24, 2)), (2, 1, 0)) # 2, 24, 2
        # self.layers[0].W = np.transpose(np.reshape(weights[0], (8, 24, 8)), (2, 1, 0))[:2, :, :2] # 2, 24, 2
        self.layers[2].W = np.transpose(np.reshape(weights[1][:4, :8], (2, 2, 8)), (2, 1, 0)) # 8, 2, 2
        # self.layers[2].W = np.transpose(np.reshape(weights[1], (2, 4, 16)), (2, 1, 0))[:8, :2, :2] # 8, 2, 2
        self.layers[4].W = np.transpose(np.reshape(weights[2], (2, -1, 4)), (2, 1, 0)) # 4, 8, 2
        # Load the weights for your CNN from the MLP Weights given
        # weights[0] = np.transpose(weights[0]) # 192, 8 -> 8, 192 = o, i*k
        # weights[0] = np.reshape(weights[0], (8, 8, 24)) # 8, 192 -> 8,8,24 = o, k, i
        # weights[0] = np.transpose(weights[0], (0, 2, 1)) # 8,8,24 -> 8, 24, 8 = o, i, k
        # weights[0] = weights[0][:2, :, :2] # 8,24,8 -> 2,24,2 = o,i,k
        #
        # weights[1] = np.transpose(weights[1]) # 8,16 -> 16, 8 = o, i*k
        # weights[1] = np.reshape(weights[1], (16, 4, 2)) # 16,8 -> 16,4,2 = o, k, i
        # weights[1] = np.transpose(weights[1], (0, 2, 1)) # 16,4,2 -> 16,2,4 = o,i,k
        # weights[1] = weights[1][:8, :, :2] # 16,2,4 -> 8, 2, 2 = o,i,k
        #
        # weights[2] = np.transpose(weights[2]) # 16, 4 -> 4, 16 = o, i*k
        # weights[2] = np.reshape(weights[2], (4, 2, 8)) # 4, 16 -> 4,2,8 = o, k, i
        # weights[2] = np.transpose(weights[2], (0, 2, 1)) # 4,2,8 -> 4,8,2 = o, i, k
        # # weights[2] = weights[2][]
        #
        # self.layers[0].W = weights[0]
        # self.layers[2].W = weights[1]
        # self.layers[4].W = weights[2]
        #

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
