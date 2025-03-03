import torch.nn as nn
import torch

from .extractors import LSTMExtractor, ConvExtractor, AttentionExtractor, ExtractorType

class Vanilla(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=10, hidden_activation=nn.Tanh(), final_activation=nn.Sigmoid(), dropout=0.3, compress=None):
        super(Vanilla, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.dropout = nn.Dropout(p=dropout)

        for i in range(layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))

        self.last_layer = nn.Linear(hidden_size, output_size)
        self.last_norm = nn.BatchNorm1d(output_size)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.last_layer.weight)


    def forward(self, input_data):
        x = self.hidden_activation(self.first_layer(input_data))
        for i in range(0, len(self.layers), 2):
            linear_layer = self.layers[i]
            batch_norm_layer = self.layers[i + 1]
            x = linear_layer(x)
            x = batch_norm_layer(x)
            x = self.hidden_activation(x)
            x = self.dropout(x)
        final_x1 = self.last_layer(x)
        final_x2 = self.last_norm(final_x1)
        predictions = self.final_activation(final_x2)
        return predictions


class VanillaWithExtractors(nn.Module):
    def __init__(self, extractor_types, input_size=None, seq_len=None,
                 hidden_size=128, output_size=1, layers=10, hidden_activation=nn.Tanh(),
                 final_activation=nn.Sigmoid(), dropout=0.3):
        super().__init__()
        self.extractor_types = extractor_types  # List of extractor types
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.dropout = nn.Dropout(p=dropout)

        if len(extractor_types) == 0:
            raise ValueError("Empty extractor types list - use Vanilla instead")

        # Require sequence dimensions when using extractors
        if seq_len is None or input_size is None:
            raise ValueError("seq_len and num_features must be provided when using extractors.")

        # Create a list of extractor instances
        self.extractors = nn.ModuleList()
        for extractor_type in extractor_types:
            if extractor_type == ExtractorType.LSTM:
                self.extractors.append(LSTMExtractor(input_size, hidden_size))
            elif extractor_type == ExtractorType.CONVOLUTION:
                self.extractors.append(ConvExtractor(input_size, hidden_size))
            elif extractor_type == ExtractorType.ATTENTION:
                self.extractors.append(AttentionExtractor(input_size, hidden_size))
            else:
                raise ValueError(f"Invalid extractor type: {extractor_type}")
        predictor_input_size = hidden_size * len(extractor_types)

        # Predictor: First layer adjusts input size to hidden_size
        self.first_layer = nn.Linear(predictor_input_size, hidden_size)

        # Stack of hidden layers
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))

        # Final output layer
        self.last_layer = nn.Linear(hidden_size, output_size)
        self.last_norm = nn.BatchNorm1d(output_size)

        # Initialize weights
        nn.init.xavier_uniform_(self.first_layer.weight)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.last_layer.weight)

    def forward(self, input_data):

        # Apply each extractor and concatenate outputs
        extractor_outputs = [extractor(input_data) for extractor in self.extractors]
        x = torch.cat(extractor_outputs, dim=1)  # (batch_size, hidden_size * num_extractors)
        x = self.hidden_activation(self.first_layer(x))

        # Pass through hidden layers
        for i in range(0, len(self.layers), 2):
            linear_layer = self.layers[i]
            batch_norm_layer = self.layers[i + 1]
            x = linear_layer(x)
            x = batch_norm_layer(x)
            x = self.hidden_activation(x)
            x = self.dropout(x)

        # Final output
        x = self.last_layer(x)
        x = self.last_norm(x)
        predictions = self.final_activation(x)
        return predictions
