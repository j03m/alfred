import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from alfred.data import LogReturnScaler


def log_return_scaler():
    # Instantiate scaler
    scaler = LogReturnScaler(cumsum=True, amplifier=2)

    # Generate fake data
    data = pd.Series([10, 12, 15, 20, 25, 30, 35, 40, 45, 50])

    # Fit and transform data
    transformed_data = scaler.fit_transform(data)

    # Inverse transform the data
    inversed_data = scaler.inverse_transform(transformed_data)
    np.set_printoptions(suppress=True)
    print(data, " VS ", inversed_data)


# Run the test
#log_return_scaler()


from sklearn.preprocessing import PowerTransformer

# Initialize the scaler
scaler = PowerTransformer(method='yeo-johnson')

# Fit and transform the data
data = np.array([[10], [12], [15], [20], [25], [30], [35], [40], [45], [50]])
transformed_data = scaler.fit_transform(data)

# Inverse transform the data
inversed_data = scaler.inverse_transform(transformed_data)

print("Transformed data:", transformed_data)
print("Inverse transformed data:", inversed_data)