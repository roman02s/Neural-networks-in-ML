import numpy as np


class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).
        self.W = np.random.randn(input_size, output_size)*0.01
        self.b = np.zeros(output_size)

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = X
        return X.dot(self.W)+self.b

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdW = self.X.T.dot(dLdy)
        self.dLdb = dLdy.sum(0)
        self.dLdx = dLdy.dot(self.W.T)
        return self.dLdx

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - learning_rate*dLdw
        '''
        self.W = self.W - learning_rate * self.dLdW
        self.b = self.b - learning_rate * self.dLdb


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.sigm_out = 1.0 / (1 + np.exp(-X))
        return self.sigm_out

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        dLdX = dLdy * self.sigm_out * (1 - self.sigm_out)
        return dLdX

    def step(self, learning_rate):
        pass


class NLLLoss:
    def __init__(self):
        pass

    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''

        self.n_samples = X.shape[0]
        self.n_classes = X.shape[1]
        self.Z = self.softmax(X)
        self.Y = self.n_log_likehood(y)
        L = self.cross_entropy(self.Z, self.Y)
        return L

    def softmax(self, X):
        max_xi = np.max(X)
        s_value = X - np.log(np.sum(np.exp(X - max_xi), axis=-1, keepdims=True)) - max_xi
        return np.exp(s_value)

    @staticmethod
    def cross_entropy(X, y):
        return -(np.log(X) * y).sum(1).mean(0)

    def n_log_likehood(self, y):
        n_log_likehood = np.zeros((self.n_samples, self.n_classes))
        n_log_likehood[np.arange(self.n_samples), y.T] = 1
        return n_log_likehood

    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        dLdx = (self.Z - self.Y) / self.Y.shape[0]
        return dLdx

class NeuralNetwork:
    def __init__(self, modules):
        '''
        Constructs network with *modules* as its layers
        '''
        self.modules = modules
    
    def forward(self, X):
        y = X
        for i in range(len(self.modules)):
            y = self.modules[i].forward(y)
        return y
    
    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        for i in range(len(self.modules))[::-1]:
            dLdy = self.modules[i].backward(dLdy)
    
    def step(self, learning_rate):
        for i in range(len(self.modules)):
            self.modules[i].step(learning_rate)
