import torch
import torch.nn as nn


class HuberWithSignPenalty(nn.Module):
    def __init__(self, delta=1.0, lambda_penalty=2.0):
        super(HuberWithSignPenalty, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.lambda_penalty = lambda_penalty

    def forward(self, predictions, labels):
        huber_loss = self.huber(predictions, labels)
        sign_mismatch = (torch.sign(predictions) != torch.sign(labels)).float()
        sign_penalty = sign_mismatch.mean()
        return huber_loss + self.lambda_penalty * sign_penalty


class ErrorAmplifiedSignLoss(nn.Module):
    def __init__(self, error_factor=2.0, lambda_penalty=2.0):
        super(ErrorAmplifiedSignLoss, self).__init__()
        self.error_factor = error_factor
        self.lambda_penalty = lambda_penalty

    def forward(self, predictions, labels):
        errors = (predictions - labels) ** 2
        sign_mismatch = (torch.sign(predictions) != torch.sign(labels)).float()
        amplified_errors = errors * (1 + self.error_factor * sign_mismatch)
        mse = amplified_errors.mean()
        sign_penalty = sign_mismatch.mean()
        return mse + self.lambda_penalty * sign_penalty


class MSEWithSignPenalty(nn.Module):
    def __init__(self, lambda_penalty=2):
        super(MSEWithSignPenalty, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, predictions, labels):
        # Compute the MSE loss
        mse = self.mse_loss(predictions, labels)

        # Compute the sign mismatch penalty
        sign_mismatch = (torch.sign(predictions) != torch.sign(labels)).float()
        sign_penalty = (sign_mismatch * torch.abs(predictions)).mean()

        # Combine MSE and sign penalty
        total_loss = mse + self.lambda_penalty * sign_penalty
        return total_loss