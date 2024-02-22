import numpy as np
import pickle
from PIL import Image

def shape_image(images):
    my_image = np.array(images).reshape(784,)
    my_image = my_image/255
    return my_image

def size_image(images):
    my_image_resized = None
    try:
        with Image.open(images) as image:
            my_image_grayscaled = image.convert("L")
            my_image_resized = my_image_grayscaled.resize((28,28))
    except Exception as e:
        print("Not able to process the image: {}".format(e))
    return my_image_resized

Trained_model = None
with open("Sai_Shanmukh_Varma_mnist_nn_model.pkl", 'rb') as model:
    Trained_model = pickle.load(model)

Test_image_size = size_image("Test_Images/9_1.png")
Test_image_shape = shape_image(Test_image_size)

y = Trained_model.predict(Test_image_shape)
print("My trained Model {}".format(y))
y_hat = np.argmax(y)
certainity = np.max(y)*100

print("Model output is predicted as {} with a certainity of {}".format(y_hat,certainity))