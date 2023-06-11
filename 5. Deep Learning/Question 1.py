"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Deep Learning Assignment 1
"""

"""
Question 1 -

Implement 3 different CNN architectures with a comparison table for the MNSIT
dataset using the Tensorflow library.

Note -

1. The model parameters for each architecture should not be more than 8000
parameters
2. Code comments should be given for proper code understanding.
3. The minimum accuracy for each accuracy should be at least 96%
"""

# Ans:

import tensorflow as tf
from tensorflow.keras import layers

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
num_classes = 10

# Define a function to create a CNN model
def create_cnn_model():
    model = tf.keras.Sequential([
        layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Model 1: CNN with 1 Convolutional layer and 1 Dense layer
model1 = create_cnn_model()
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.summary()

# Train model 1
model1.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)

# Evaluate model 1
test_loss1, test_acc1 = model1.evaluate(x_test, y_test, verbose=0)
print(f"Model 1 - Test accuracy: {test_acc1*100:.2f}%")

# Model 2: CNN with 2 Convolutional layers and 1 Dense layer
model2 = tf.keras.Sequential([
    layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model2.summary()

# Train model 2
model2.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)

# Evaluate model 2
test_loss2, test_acc2 = model2.evaluate(x_test, y_test, verbose=0)
print(f"Model 2 - Test accuracy: {test_acc2*100:.2f}%")

# Model 3: CNN with 3 Convolutional layers and 1 Dense layer
model3 = tf.keras.Sequential([
    layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])
model3.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model3.summary()

# Train model 3
model3.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)

# Evaluate model 3
test_loss3, test_acc3 = model3.evaluate(x_test, y_test, verbose=0)
print(f"Model 3 - Test accuracy: {test_acc3*100:.2f}%")

# Comparison table
print("\nComparison Table:")
print("-------------------------------------------------")
print("Model\t\t| Parameters\t| Test Accuracy")
print("-------------------------------------------------")
print(f"Model 1\t\t| {model1.count_params()}\t\t| {test_acc1*100:.2f}%")
print(f"Model 2\t\t| {model2.count_params()}\t\t| {test_acc2*100:.2f}%")
print(f"Model 3\t\t| {model3.count_params()}\t\t| {test_acc3*100:.2f}%")
print("-------------------------------------------------")