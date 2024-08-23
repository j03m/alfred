import torch
def calculate_relative_error_percentage(mse, target_tensor):
    # Calculate the square root of the MSE to get the average error magnitude
    avg_error_magnitude = torch.sqrt(mse)

    # Calculate the range of the target variable
    min_value = torch.min(target_tensor)
    max_value = torch.max(target_tensor)
    target_range = max_value - min_value

    # Calculate the relative error
    relative_error = avg_error_magnitude / target_range

    # Convert to percentage
    return relative_error.item() * 100

def calculate_robust_relative_error_percentage(mse, target_tensor, prediction_tensor):
    # Calculate the square root of the MSE to get the average error magnitude
    avg_error_magnitude = torch.sqrt(mse)

    # Calculate the range of the target variable
    min_target = torch.min(target_tensor)
    max_target = torch.max(target_tensor)
    target_range = max_target - min_target

    # Calculate the range of the prediction values
    min_pred = torch.min(prediction_tensor)
    max_pred = torch.max(prediction_tensor)
    pred_range = max_pred - min_pred

    # Calculate the combined range (target + prediction)
    combined_range = (target_range + pred_range) / 2

    # Calculate the robust relative error
    robust_relative_error = avg_error_magnitude / combined_range

    # Convert to percentage
    return robust_relative_error.item() * 100

# # Example usage
# mse = torch.tensor(0.029)  # Your MSE value
# target_tensor = torch.tensor([-0.5185, 1.4151])  # Example target values
# prediction_tensor = torch.tensor([-0.4000, 1.3500])  # Example predicted values
#
# calculate_robust_relative_error_percentage(mse, target_tensor, prediction_tensor)
