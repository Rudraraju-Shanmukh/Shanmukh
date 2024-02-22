import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

epochs = 5
class DogsCatsAug:
    def __init__(self):
        self.data_from_kaggle = "data_from_kaggle/train"
        self.data_dirname = "dogs_vs_cats"
        self.model = self.build_model()

    def build_model(self):
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ])
        inputs = tf.keras.Input(shape=(180, 180, 3))
        augmented_inputs = data_augmentation(inputs)
        preprocessed_inputs = layers.Rescaling(1./255)(augmented_inputs)
        x = layers.Conv2D(32, (3, 3), activation='relu')(preprocessed_inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def train(self, model_name, train_dataset, validation_dataset):
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        history = self.model.fit(train_dataset,
                                 validation_data=validation_dataset,
                                 epochs= epochs,
                                 callbacks=[checkpoint_cb])
        return history

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def prepare_datasets(self):
        train_dataset = image_dataset_from_directory(os.path.join(self.data_dirname, 'train'),
                                                     image_size=(180, 180),
                                                     batch_size=32)
        validation_dataset = image_dataset_from_directory(os.path.join(self.data_dirname, 'validation'),
                                                          image_size=(180, 180),
                                                          batch_size=32)
        test_dataset = image_dataset_from_directory(os.path.join(self.data_dirname, 'test'),
                                                    image_size=(180, 180),
                                                    batch_size=32)
        return train_dataset, validation_dataset, test_dataset

    def fit(self, train_dataset, validation_dataset):
        history = self.model.fit(train_dataset, validation_data=validation_dataset, epochs= epochs)
        return history

    def predict(self, model_name, file_name):
        model = tf.keras.models.load_model(model_name)
        img = tf.keras.utils.load_img(file_name, target_size=(180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        return predictions[0][0]
