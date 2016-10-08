"""Template Based Learning on MNIST Dataset

This model uses a matrix of 28x28 for each class (digits from 0 to 9),
and in each training examples, increment the overlap pixels of correct class,
while decrement the overlap pixels of incorrect classes
"""

# Libraries
import numpy as np
from keras.datasets import mnist

# Matrices for each class
matrices = np.zeros((10, 28, 28), dtype=np.int32)

# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# For each sample
for i in range(len(X_train)):
  # Get `X` and `y`
  X, y = X_train[i], y_train[i]

  # Get the mask
  mask = (X > 0).astype(np.int32)
  
  # Increment the correct label
  matrices[y] += mask

  # Decrement
  for f_y in range(10):
    if f_y != y:
      matrices[f_y] += -mask

# Test
score = 0
for i in range(len(X_test)):
  # Get `X` and `y`
  X, y = X_test[i], y_test[i]

  # Get the score for each class
  scores = [ np.sum(X * matrices[v]) for v in range(10) ]
  predicted_y = np.argmax(scores)

  if predicted_y == y:
    score += 1

# Should give about 66,13% of accuracy
print("Final score: {}".format(score / len(X_test)))