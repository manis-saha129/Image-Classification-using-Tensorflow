import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets
import matplotlib.pyplot as plt
import random

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Define the class labels for CIFAR-10
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Number of images to display
num_images_to_display = 25

# Create a figure to display the images
plt.figure(figsize=(10, 10))

# Randomly select and display images
for i in range(num_images_to_display):
    index = random.randint(0, x_test.shape[0] - 1)
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_test[index])
    plt.title(class_labels[y_test[index][0]])
    plt.axis('off')

plt.tight_layout()
plt.show()
