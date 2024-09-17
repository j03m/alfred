import numpy as np
import matplotlib.pyplot as plt


def get_positional_encodings(max_len, d_model):
    # Initialize the positional encoding matrix
    pe = np.zeros((max_len, d_model))

    # Get the position indices
    position = np.arange(0, max_len).reshape(-1, 1)  # Shape (max_len, 1)

    # Calculate the div_term for different dimensions
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    # Apply sine to even indices in the encoding
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices in the encoding
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


# Let's test this function for a sequence of length 100 and a model with dimension 16
pe = get_positional_encodings(100, 16)
print(pe.shape)  # Should be (100, 16)


def plot_positional_encodings(pe, positions):
    plt.figure(figsize=(12, 6))

    for pos in range(0, positions):
        plt.plot(pe[pos], label=f"Position {pos}")

    plt.xlabel("Dimension")
    plt.ylabel("Positional Encoding Value")
    plt.title("Positional Encodings for Different Positions")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot positional encodings for positions 0, 10, 50, and 99
plot_positional_encodings(pe, 100)

def compare_positional_encodings(pe, pos1, pos2):
    plt.figure(figsize=(8, 4))
    plt.plot(pe[pos1], label=f"Position {pos1}")
    plt.plot(pe[pos2], label=f"Position {pos2}", linestyle='dashed')
    plt.xlabel("Dimension")
    plt.ylabel("Positional Encoding Value")
    plt.title(f"Comparison of Encodings: Position {pos1} vs {pos2}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Compare the encodings for position 10 and 20
compare_positional_encodings(pe, 10, 20)


def simple_scalar_encoding(max_len, d_model):
    # Just create a linear ramp of positions
    return np.tile(np.arange(max_len).reshape(-1, 1), (1, d_model))

# Generate simple scalar encodings and compare them
simple_pe = simple_scalar_encoding(100, 16)

plot_positional_encodings(simple_pe, 100)