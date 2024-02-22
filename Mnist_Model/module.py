import numpy as np
import matplotlib.pyplot as plt
from mnist import Mnist  

# Create an instance of the Mnist class
mnist = Mnist()

# Load the neural network model
mnist.init_network()

# Load test images and labels
x_test = mnist.load_images(mnist.key_file['test_img'])
y_test = mnist.load_labels(mnist.key_file['test_label'])

# Select a prediction image for prediction
input_image = x_test[1000]/255

# Perform a prediction
y = mnist.predict(input_image)
y_hat = np.argmax(y)
confidence = np.max(y) * 100

# Display the predict image and prediction
plt.imshow(input_image.reshape(28, 28), cmap='gray')
plt.title(f"input image : {np.argmax(y)}")
plt.show()

if y_hat == y_test[1000]:
    print('success')
else:
    print('fail')

print(f'x[1000] is predicted as {y_hat} with {confidence}%. The label is {y_test[1000]}')

