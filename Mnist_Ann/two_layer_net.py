import numpy as np
import common
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = common.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        y = common.softmax(a2)

        return y
     
    def loss(self, x, t):
        y = self.predict(x)
        return common.cross_entropy_error(y, t)
    
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        grads['w1'] = common.numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = common.numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = common.numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = common.numerical_gradient(loss_w, self.params['b2'])

        return grads
