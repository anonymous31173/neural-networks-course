"""Perceptron Implementation using numpy."""

# Numerical Libraries
import numpy as np
from matplotlib import pyplot, pylab

class Perceptron(object):
  def __init__(self, number_of_samples, dimensions=2):
    # The number of input dimensions
    self.input_dimensions = 2

    # Initialize the weights
    self.weights = np.random.randn(self.input_dimensions + 1) # add the bias unit

    # Max number of iterations
    self.max_iter = 200

    # Learning rate
    self.eta = 0.1

  def fit(self, X, y):
    # Flag to check if converged
    converged = False

    # Actual iteration
    actual_iter = 0

    # While not converged or less then `max_iter`
    while not converged or actual_iter <= self.max_iter:
      # Go to each row of the dataset
      for x_sample, y_sample in zip(X, y):
        # The the current output
        # z = W.T * X
        z = self.weights.dot(x_sample)
        z_binary = z >= 0

        # If the outputs doesn't match
        if z_binary != y_sample:
          # Calculate the new weights
          self.weights += self.eta * x_sample * (y_sample - z_binary)

          # Set that didn't converged
          converged = False
        else:
          converged = True
        actual_iter += 1