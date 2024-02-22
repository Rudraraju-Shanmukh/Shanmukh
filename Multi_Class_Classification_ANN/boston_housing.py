
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import boston_housing

class BostonHousing:
    def __init__(self):
        self.model = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.normalized_train_data = None
        self.normalized_test_data = None

    def load_and_normalize_data(self):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = boston_housing.load_data()
        self.normalized_train_data = self.normalize_data(self.train_data)
        self.normalized_test_data = self.normalize_data(self.test_data)

    def normalize_data(self, data):
        mean = data.mean(axis=0)
        data -= mean
        std = data.std(axis=0)
        data /= std
        return data

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_shape=(self.normalized_train_data.shape[1],)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer='ADAM', loss='mse', metrics=['mae'])
    
    def train_model(self, epochs=200, batch_size=32, validation_split=0.2):
        history = self.model.fit(self.normalized_train_data, self.train_labels, 
                                 validation_split=validation_split, 
                                 batch_size=batch_size, epochs=epochs)
        return history

    def plot_training_history(self, history):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'r-', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b--', label='Validation Loss')
        plt.title('Training vs. Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
