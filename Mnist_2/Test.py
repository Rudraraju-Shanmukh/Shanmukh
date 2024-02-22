from mnist_keras import MnistKeras

(_,_), (x_test, y_test) = MnistKeras.load_data() #importing the load data from MnistKeras class
model = MnistKeras.load("Model_Sai_Shanmukh_Varma_Rudraraju") # Loading the trained model
MnistKeras.test(model)
