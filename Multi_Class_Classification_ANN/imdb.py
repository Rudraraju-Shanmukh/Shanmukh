
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import imdb

class Imdb:
    def __init__(self):
        self.NUM_WORDS = 10000
        self.NUM_EPOCHS = 30
        self.BATCH_SIZE = 512
        self.VALIDATION_SPLIT = 0.2
        self.PATIENCE = 5
        self.model = None

    def load_data(self):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words=self.NUM_WORDS)
        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)
        self.y_train = np.asarray(self.train_labels).astype('float32')
        self.y_test = np.asarray(self.test_labels).astype('float32')

    def vectorize_sequences(self, sequences):
        results = np.zeros((len(sequences), self.NUM_WORDS))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(16, activation='relu', input_shape=(self.NUM_WORDS,)))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(optimizer='ADAM', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
        history = self.model.fit(self.x_train, self.y_train, epochs=self.NUM_EPOCHS, 
                                 batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT, 
                                 callbacks=[callback])
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

    def save_model(self, filename):
        self.model.save(filename)

    def predict(self, data):
        return self.model.predict(data)

    def test_new_input(self, new_data):
        new_data_vec = self.vectorize_sequences([new_data])
        return self.model.predict(new_data_vec)
