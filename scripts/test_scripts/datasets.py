from next_gen import SlidingWindowDerivedOutputDataset
import pandas as pd
import torch

data = {
    'feature1': range(1, 21),
    'feature2': range(21, 41),
    'price': range(101, 121)
}

df = pd.DataFrame(data)
feature_columns = ['feature1', 'feature2']
input_window = 3
derivations = [1, 2]
derivation_column = 'price'

# Create an instance of the dataset
dataset = SlidingWindowDerivedOutputDataset(df, feature_columns, input_window, derivations, derivation_column)

# Test the length of the dataset
expected_length = len(df) - input_window - max(derivations)
assert len(dataset) == expected_length, f"Expected length {expected_length}, but got {len(dataset)}"
print("Length of the dataset:", len(dataset))

# Test accessing the first item
first_input, first_output = dataset[0]

# given an input window of 3 feature 1 ranges from 1-3 and feature 2 ranges from 21-23
expected_first_input = torch.tensor([[1.0, 21.0], [2.0, 22.0], [3.0, 23.0]])

# this is the expected first profit. Given derivations of 1 and 2 should expect that the
# first profit bar is the last price in the input window (3) subtraced from the price at derivations away.
# given a price range start of 101, the last input window price should be 103 and the first dervation bar 104
# yielding an output of 1. The next derivation is 2, so 2 bars from the last price would be 105 yield profit of 2
expected_first_output = torch.tensor([1.0, 2.0])
assert torch.equal(first_input, expected_first_input)


assert torch.equal(first_output, expected_first_output)

# Test accessing the last item
last_index = len(dataset) - 1
last_item = dataset[last_index]

# the last pieces of input should be the last available window, which is going to our last possible piece of data
# but subtacted from that index is the input and output windows. Our output window is our max derivation.
# so given an input window of 3 and a max derivation of 2, our first price in our last window given a len of 20
# (range 1-21) should be at position 15. which would be value 15. Follwing that the range 21-41 should land on 35
#
expected_last_input = torch.tensor([[15.0, 35.0], [16.0, 36.0], [17.0, 37.0]])
# derivation window and price increments are the same so profit is the same
expected_last_output = torch.tensor([1.0, 2.0])
assert torch.equal(last_item[0], expected_last_input), f"Expected last input {expected_last_input}, but got {last_item[0]}"
assert torch.equal(last_item[1], expected_last_output), f"Expected last output {expected_last_output}, but got {last_item[1]}"
print("Last item:", last_item)