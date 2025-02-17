import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Create Two Sample Datasets (Simulating Different CSVs)
np.random.seed(0)  # for reproducibility

# Dataset 1: Base dataset (e.g., initial 100 equities)
data1 = pd.DataFrame({
    'Feature1': np.random.normal(loc=50, scale=10, size=1000), # Mean 50, Std Dev 10
    'Dataset': 'Dataset 1'
})

# Dataset 2: Dataset with a distribution shift (e.g., a new equity with different characteristics)
data2 = pd.DataFrame({
    'Feature1': np.random.normal(loc=70, scale=10, size=1000), # Mean shifted to 70, Std Dev 10
    'Dataset': 'Dataset 2'
})

combined_data = pd.concat([data1, data2])

# 2. Visualize Original Data Distributions (Before Scaling)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=combined_data, x='Feature1', hue='Dataset', kde=True)
plt.title('Original Feature Distributions (Before Scaling)')

# 3. Demonstrate Inconsistent Scaling (Scale Each Dataset Separately)
scaler1 = StandardScaler()
data1_scaled_inconsistent = scaler1.fit_transform(data1[['Feature1']])
data1_scaled_inconsistent = pd.DataFrame({'Feature1_Scaled_Inconsistent': data1_scaled_inconsistent.flatten(), 'Dataset': 'Dataset 1'})

scaler2 = StandardScaler()
data2_scaled_inconsistent = scaler2.fit_transform(data2[['Feature1']])
data2_scaled_inconsistent = pd.DataFrame({'Feature1_Scaled_Inconsistent': data2_scaled_inconsistent.flatten(), 'Dataset': 'Dataset 2'})

combined_scaled_inconsistent = pd.concat([data1_scaled_inconsistent, data2_scaled_inconsistent])

plt.subplot(1, 2, 2)
sns.histplot(data=combined_scaled_inconsistent, x='Feature1_Scaled_Inconsistent', hue='Dataset', kde=True)
plt.title('Inconsistent Scaling: Distributions Still Shifted')
plt.tight_layout()
plt.show()


# 4. Demonstrate Consistent Scaling (Scale Using Parameters from Dataset 1)
scaler_consistent = StandardScaler()
scaler_consistent.fit(data1[['Feature1']]) # Fit scaler ONLY on Dataset 1

data1_scaled_consistent = scaler_consistent.transform(data1[['Feature1']]) # Transform Dataset 1
data1_scaled_consistent = pd.DataFrame({'Feature1_Scaled_Consistent': data1_scaled_consistent.flatten(), 'Dataset': 'Dataset 1'})

data2_scaled_consistent = scaler_consistent.transform(data2[['Feature1']]) # Transform Dataset 2 using SAME scaler
data2_scaled_consistent = pd.DataFrame({'Feature1_Scaled_Consistent': data2_scaled_consistent.flatten(), 'Dataset': 'Dataset 2'})

combined_scaled_consistent = pd.concat([data1_scaled_consistent, data2_scaled_consistent])


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=combined_data, x='Feature1', hue='Dataset', kde=True)
plt.title('Original Feature Distributions (Before Scaling)') # Replot Original for Comparison

plt.subplot(1, 2, 2)
sns.histplot(data=combined_scaled_consistent, x='Feature1_Scaled_Consistent', hue='Dataset', kde=True)
plt.title('Consistent Scaling: Shift Maintained, Scales Consistent')
plt.tight_layout()
plt.show()