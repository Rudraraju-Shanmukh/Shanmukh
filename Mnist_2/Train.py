from mnist_keras import MnistKeras

(x_train, y_train), (_,_) = MnistKeras.load_data() #importing the load data from MnistKeras class
model = MnistKeras.build_model()
model = MnistKeras.train(model, "Model_Sai_Shanmukh_Varma_Rudraraju") # Saving the model
