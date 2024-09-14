import numpy as np

# Fake data simulation
np.random.seed(42)
data = np.random.randn(20)  # Simulating 20 days of stock prices

seq_length = 5
change = 1  # Number of days ahead for the target

# Creating x using np.lib.stride_tricks.as_strided
n_row = data.shape[0] - seq_length + 1
x = np.lib.stride_tricks.as_strided(data, shape=(n_row, seq_length),
                                    strides=(data.strides[0], data.strides[0]))
x = np.expand_dims(x[:-1], 2)  # Expand dimensions to simulate 3D structure for model

# Creating y as the price 'change' days ahead of the last day in the sequence
y = data[seq_length + change - 1:]

# Displaying x and y for verification
print(data)
print(x)
print(y)