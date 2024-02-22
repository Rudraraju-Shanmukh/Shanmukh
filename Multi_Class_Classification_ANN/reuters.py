
import numpy as np
from keras import models
from keras import layers
from keras.datasets import reuters
from keras.utils import to_categorical 
import matplotlib.pyplot as plt

class Reuters:
    def __init__(self):
        self.NUM_WORDS = 10000
        self.NUM_EPOCHS = 20
        self.BATCH_SIZE = 512
        self.model = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

    def load_data(self):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = reuters.load_data(num_words=self.NUM_WORDS)
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)
        self.one_hot_train_labels = to_categorical(self.train_labels)
        self.one_hot_test_labels = to_categorical(self.test_labels)

    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.NUM_WORDS))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_shape=(self.NUM_WORDS,)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(46, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        history = self.model.fit(self.x_train, self.one_hot_train_labels, epochs=self.NUM_EPOCHS, 
                                 batch_size=self.BATCH_SIZE, validation_split=0.2)
        return history

    def plot_training_history(self, history):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'r-', label='training loss')
        plt.plot(epochs, val_loss_values, 'b--', label='validation loss')
        plt.title('Training vs. Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        return self.model.evaluate(self.x_test, self.one_hot_test_labels)

    def predict(self, data):
        return self.model.predict(self.vectorize_sequences([data]))
