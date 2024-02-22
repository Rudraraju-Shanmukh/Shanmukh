import numpy as np
#Calling the MLP class
from multilayer_perceptron import MLP
w1 = np.array([[0.2,0.4,0.6],[0.1,0.3,0.5]])
b1 = np.array([0.1,0.2,0.3])
w2 = np.array([[0.2,0.4],[0.3,0.4],[0.3,0.5]])
b2 = np.array([0.1,0.3])
w3 = np.array([[0.4,0.6],[0.1,0.5]])
b3 = np.array([0.2,0.2])
#Assigning the values to the  model
mlp = MLP(w1,b1,w2,b2,w3,b3)

#Input 1
input1 = np.array([4,6])
output1 = mlp.model(input1)
print(f"Input 1 Z1,Z2,Z3 values are {output1}")

#Input 2
input2 = np.array([3,5])
output2 = mlp.model(input2)
print(f"Input 2 Z1,Z2,Z3 values are {output2}")

#Input 3
input3 = np.array([5,7])
output3 = mlp.model(input3)
print(f"Input 3 Z1,Z2,Z3 values are {output3}")