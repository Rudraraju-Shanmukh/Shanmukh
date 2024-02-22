import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

class MnistKeras:

    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float32') / 255 # Normalizing the Train Data
        x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255 # Normalizing the Test Data
        
        y_train = to_categorical(y_train) # Coverting Values to 0-1
        y_test = to_categorical(y_test)   # Coverting Values to 0-1
        return (x_train, y_train), (x_test, y_test)


    def build_model():
        # Build the Sequential model
        model = Sequential()
        model.add(Dense(100, input_dim=28*28, activation='relu')) # First Layer
        model.add(Dense(10, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compiling the model
        return model


    def train(model, model_name):
        (x_train, y_train), (_,_) = MnistKeras.load_data()
        model.fit(x_train, y_train, epochs=20, batch_size=200, verbose=1) # Fitting the model
        model.save(model_name) # Saving the model
        return model


    def load(model_name):
        return load_model(model_name)

    def test(model):
        (_,_), (x_test, y_test) = MnistKeras.load_data()
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Test Loss: {}".format(loss)) # Print Test Loss
        print("Test Accuracy: {}".format(accuracy)) # Print Test Accuracy


    def predict(model, x):
        return model.predict(x) # predicting the output
