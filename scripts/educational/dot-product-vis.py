import numpy as np
import matplotlib.pyplot as plt

# Define parallel vectors
A = np.array([3, 4])
B_same = np.array([6, 8])  # Same direction
B_opposite = np.array([-3, -4])  # Opposite direction

# Calculate dot products
dot_same = np.dot(A, B_same)
dot_opposite = np.dot(A, B_opposite)

# Plot the vectors
origin = [0], [0]
# plt.quiver(*origin, A[0], A[1], color='r', angles='xy', scale_units='xy', scale=1, label='A')
# plt.quiver(*origin, B_same[0], B_same[1], color='b', angles='xy', scale_units='xy', scale=1, label=f'B (Same Dir), Dot={dot_same}')
# plt.quiver(*origin, B_opposite[0], B_opposite[1], color='g', angles='xy', scale_units='xy', scale=1, label=f'B (Opp Dir), Dot={dot_opposite}')

# Plot projections of A onto B_same and B_opposite
proj_B_same = (np.dot(A, B_same) / np.dot(B_same, B_same)) * B_same
proj_B_opposite = (np.dot(A, B_opposite) / np.dot(B_opposite, B_opposite)) * B_opposite

# Projections as dashed lines
plt.plot([0, proj_B_same[0]], [0, proj_B_same[1]], 'c--', label='Projection on B (Same Dir)')
plt.plot([0, proj_B_opposite[0]], [0, proj_B_opposite[1]], 'm--', label='Projection on B (Opp Dir)')

# Plot settings
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('Visualization of Parallel Vectors and Dot Products')
plt.show()