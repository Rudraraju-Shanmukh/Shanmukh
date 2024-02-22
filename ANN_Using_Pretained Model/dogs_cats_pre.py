import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import os

class DogsCats:
    def __init__(self):
        self.model = self.build_model()
        self.model_name = None 

    def build_model(self):

        data_augmentation = tf.keras.Sequential( [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2)])

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = layers.Flatten()(base_model.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        return model

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def prepare_datasets(self, dataset_directory):
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dataset_directory, 'train'),
            image_size=(224, 224), 
            batch_size=32)

        validation_dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dataset_directory, 'validation'),
            image_size=(224, 224),  
            batch_size=32)

        test_dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(dataset_directory, 'test'),
            image_size=(224, 224), 
            batch_size=32)

        return train_dataset, validation_dataset, test_dataset

    def fit(self, train_dataset, validation_dataset, model_name, epochs=10):
        self.model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
        self.model.save(model_name)

    def predict(self, model_name, file_name):
        model = models.load_model(model_name)
        image = tf.keras.preprocessing.image.load_img(file_name, target_size=(224, 224))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = tf.expand_dims(input_arr, 0) 
        predictions = model.predict(input_arr)
        predicted_class = "cat" if predictions[0] < 0.5 else "dog"
        print(f'The predicted image is a {predicted_class}')
