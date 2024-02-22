from mnist_data import MnistData
import two_layer_net
import numpy as np
import pickle

mnist_data = MnistData()
(x_train, t_train), (x_test, t_test) = mnist_data.load()

iters_num = 30
train_size = x_train.shape[0]
batch_size = 64
learning_rate = 0.01
train_loss = []
input_size = 28*28
Sai_Shanmukh_Varma_mnist_nn_model = two_layer_net.TwoLayerNet(input_size=input_size, hidden_size=100, output_size=10)

for i in range(iters_num):
    print("Current iteration is {}".format(i))
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = Sai_Shanmukh_Varma_mnist_nn_model.numerical_gradient(x_batch, t_batch)
    for key in ('w1', 'b1', 'w2', 'b2'):
        Sai_Shanmukh_Varma_mnist_nn_model.params[key] -= learning_rate*grad[key]
loss = Sai_Shanmukh_Varma_mnist_nn_model.loss(x_batch, t_batch)
train_loss.append(loss)

with open('Sai_Shanmukh_Varma_mnist_nn_model.pkl', 'wb') as model:
    pickle.dump(Sai_Shanmukh_Varma_mnist_nn_model,model)
print("Pickle file completed")