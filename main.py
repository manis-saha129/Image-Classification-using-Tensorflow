import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets
import matplotlib.pyplot as plt
import random

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Plot training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# # Load the CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#
# # Define the class labels for CIFAR-10
# class_labels = [
#     'airplane', 'automobile', 'bird', 'cat', 'deer',
#     'dog', 'frog', 'horse', 'ship', 'truck'
# ]
#
# # Number of images to display
# num_images_to_display = 25
#
# # Create a figure to display the images
# plt.figure(figsize=(10, 10))
#
# # Randomly select and display images
# for i in range(num_images_to_display):
#     index = random.randint(0, x_test.shape[0] - 1)
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(x_test[index])
#     plt.title(class_labels[y_test[index][0]])
#     plt.axis('off')
#
# plt.tight_layout()
# plt.show()
