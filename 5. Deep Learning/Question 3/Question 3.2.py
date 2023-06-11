"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Deep Learning Assignment 3.2
"""

"""
Question 3 -

Train a Pure CNN with less than 10000 trainable parameters using the MNIST
Dataset having minimum validation accuracy of 99.40%
Note -
1. Code comments should be given for proper code understanding.
2. Implement in both PyTorch and Tensorflow respectively
"""

# Ans:

# Tensorflow Implementation   

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10),
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")