import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def nrmse_by_range(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    range_y = np.max(y_true) - np.min(y_true)
    return rmse / range_y


# Low Range Example (1 to 3)
# Generate data where predictions are very close to actual values
actual_low = np.random.randint(1, 4, size=100)  # True ranks from 1 to 3
predictions_low = actual_low + np.random.normal(0, 0.5, 100)  # Add small noise
predictions_low_clipped = np.clip(np.round(predictions_low), 1, 3)  # Ensure predictions are within range

mse_low = mean_squared_error(actual_low, predictions_low_clipped)
print(f"Low Range MSE (1-3): {mse_low:.4f}")
print(f"NRMSE: {nrmse_by_range(actual_low, predictions_low_clipped)}")

# Wide Range Example (1 to 1000)
# Generate data where predictions are also very close but on a much larger scale
actual_wide = np.random.randint(1, 1001, size=100)  # True values from 1 to 1000
predictions_wide = actual_wide + np.random.normal(0, 0.5, 100)  # Add noise proportional to range
predictions_wide_clipped = np.clip(np.round(predictions_wide), 1, 1000)  # Ensure predictions are within range

mse_wide = mean_squared_error(actual_wide, predictions_wide_clipped)
print(f"Wide Range MSE (1-1000): {mse_wide:.4f}")
print(f"NRMSE: {nrmse_by_range(actual_wide, predictions_wide_clipped)}")

# Visualize a sample of data to see how close predictions are to actual values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(actual_low[:10])), actual_low[:10], color='blue', label='Actual')
plt.scatter(range(len(predictions_low[:10])), predictions_low[:10], color='red', label='Predicted')
plt.title('Low Range (1-3)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(actual_wide[:10])), actual_wide[:10], color='blue', label='Actual')
plt.scatter(range(len(predictions_wide[:10])), predictions_wide[:10], color='red', label='Predicted')
plt.title('Wide Range (1-1000)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()