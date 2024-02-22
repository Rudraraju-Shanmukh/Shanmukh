import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

epochs = 5
data_from_kaggle = "data_from_kaggle/train"
data_dirname = "dogs_vs_cats"

def make_dataset(subset_name, start_idx, end_idx):
    for category in {"cat", "dog"}:
        directory = f"{data_dirname}/{subset_name}/{category}"
        os.makedirs(directory, exist_ok=True)
        filenames = [f"{category}.{i}.jpg" for i in range(start_idx, end_idx)]
        for filename in filenames: 
            shutil.copyfile(src=f"{data_from_kaggle}/{filename}", dst=f"{directory}/{filename}")

class DogsCats:
    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def compile(self, compiled_model):
        compiled_model.compile(loss='binary_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])
        return compiled_model

    def prepare_datasets(self):
        train_dataset = image_dataset_from_directory(f"{data_dirname}/train",
                                                     image_size=(180, 180),
                                                     batch_size=32)
        validation_dataset = image_dataset_from_directory(f"{data_dirname}/validation",
                                                          image_size=(180, 180),
                                                          batch_size=32)
        test_dataset = image_dataset_from_directory(f"{data_dirname}/test",
                                                    image_size=(180, 180),
                                                    batch_size=32)
        return train_dataset, validation_dataset, test_dataset

    def train(self, training_model, model_name, training_dataset, validation_dataset):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)
        training_history = training_model.fit(training_dataset,
                                              validation_data=validation_dataset,
                                              epochs= epochs,
                                              callbacks=[checkpoint_callback])
        return training_history

    def predict(self, prediction_model, image_file):
        loaded_model = tf.keras.models.load_model(prediction_model)
        img = tf.keras.utils.load_img(image_file, target_size=(180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        predictions = loaded_model.predict(img_array)
        return predictions[0][0]
