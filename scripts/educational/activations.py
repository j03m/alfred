import torch
import torch.nn as nn

# Softmax expects a tensor, not a scalar
softmax = nn.Softmax(dim=0)
input_tensor = torch.tensor([0.05])  # or [0.05, -0.05] for comparison
result = softmax(input_tensor)
print("Softmax result:", result)

# For comparison with multiple values
input_tensor_multiple = torch.tensor([0.05, -0.05])
result_multiple = softmax(input_tensor_multiple)
print("Softmax result (multiple):", result_multiple)

# ReLU
relu = nn.ReLU()
print("ReLU(0.05):", relu(torch.tensor(0.05)))
print("ReLU(-0.05):", relu(torch.tensor(-0.05)))

# LeakyReLU
leaky_relu = nn.LeakyReLU(0.01)  # 0.01 is the negative slope
print("LeakyReLU(0.05):", leaky_relu(torch.tensor(0.05)))
print("LeakyReLU(-0.05):", leaky_relu(torch.tensor(-0.05)))

# Tanh
tanh = nn.Tanh()
print("Tanh(0.05):", tanh(torch.tensor(0.05)))
print("Tanh(-0.05):", tanh(torch.tensor(-0.05)))

# Sigmoid
sigmoid = nn.Sigmoid()
print("Sigmoid(0.05):", sigmoid(torch.tensor(0.05)))
print("Sigmoid(-0.05):", sigmoid(torch.tensor(-0.05)))