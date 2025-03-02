import torch.nn as nn

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
        final_x = self.last_layer(x)
        predictions = self.final_activation(final_x)
        return predictions


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


class VanillaCompress(nn.Module):
    def __init__(self, input_size, sizes, output_size, hidden_activation=nn.Tanh(), final_activation=nn.Sigmoid(), dropout=0.3):
        super(VanillaCompress, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        self.first_layer = nn.Linear(input_size, sizes[0])
        self.hidden_activation = hidden_activation
        self.final_activation = final_activation
        self.dropout = nn.Dropout(p=dropout)

        for i in range(1, len(sizes)):
            self.layers.append(nn.Linear(sizes[i-1], sizes[i]))
            self.layers.append(nn.BatchNorm1d(sizes[i]))

        for i in range(len(sizes)-1, 0, -1):
            self.layers.append(nn.Linear(sizes[i], sizes[i-1]))
            self.layers.append(nn.BatchNorm1d(sizes[i-1]))

        self.last_layer = nn.Linear(sizes[0], output_size)
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
        final_x = self.last_layer(x)
        predictions = self.final_activation(final_x)
        return predictions

