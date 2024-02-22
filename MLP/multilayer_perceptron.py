import numpy as np
class MLP:
    def __init__(self,w1,b1,w2,b2,w3,b3):
        self.nn ={}
        self.nn['w1'] = w1
        self.nn['b1'] = b1

        self.nn['w2'] = w2
        self.nn['b2'] = b2

        self.nn['w3'] = w3
        self.nn['b3'] = b3

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def model(self,x):
        w1,w2,w3 = self.nn['w1'], self.nn['w2'], self.nn['w3']
        b1,b2,b3 = self.nn['b1'], self.nn['b2'], self.nn['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1,w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2,w3) + b3
        z3 = self.sigmoid(a3)

        return z1,z2,z3