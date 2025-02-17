import numpy as np

# Generate an array with extreme outliers
data = np.array([1, 2, 3, 4, 5, 1000, 1500])

# Calculate arithmetic mean
arithmetic_mean = np.mean(data)
print(f"Arithmetic Mean: {arithmetic_mean:.2f}")

# Calculate harmonic mean
harmonic_mean = len(data) / np.sum(1/data)
print(f"Harmonic Mean: {harmonic_mean:.2f}")

print("Data:", data)