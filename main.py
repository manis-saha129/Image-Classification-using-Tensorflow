import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, datasets
import matplotlib.pyplot as plt
import cv2

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

model.summary()

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

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Function to preprocess and predict image class
def predict_and_plot(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (32, 32))
    img_normalized = img_resized / 255.0
    test_input = np.expand_dims(img_normalized, axis=0)
    predictions = model.predict(test_input)
    predicted_class = class_names[np.argmax(predictions)]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {predicted_class}')
    plt.axis('off')
    plt.show()


# Test the model with images
predict_and_plot('images/airplane.jpg')
predict_and_plot('images/automobile.jpg')
predict_and_plot('images/bird.jpg')
predict_and_plot('images/cat.jpg')
predict_and_plot('images/deer.jpeg')
predict_and_plot('images/dog.jpg')
predict_and_plot('images/frog.jpeg')
predict_and_plot('images/horse.jpeg')
predict_and_plot('images/ship.jpg')
predict_and_plot('images/truck.jpg')
