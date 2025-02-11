import numpy as np

# Your predictions as provided
predictions = [
    3.478180408477783, 1.900294303894043, 1.307655930519104, 2.5426111221313477, 2.5434396266937256,
    1.563638687133789, 1.7438769340515137, 2.639838933944702, 2.297264814376831, 2.693908214569092,
    1.4831829071044922, 2.4049758911132812, 2.8218021392822266, 2.1012303829193115, 1.7021156549453735,
    2.5162835121154785, 2.478376865386963, 1.6912555694580078, 2.198033094406128, 2.4217755794525146,
    1.9876199960708618, 1.883253812789917, 2.663939952850342, 2.1050190925598145, 2.3566861152648926,
    2.7068755626678467, 1.6688040494918823, 1.8525769710540771, 1.722890853881836, 3.0985565185546875,
    0.23327308893203735, 2.544776439666748, 3.786829948425293, 0.6293522119522095, 4.085860252380371,
    1.926673173904419
]

# Convert to numpy array for easier manipulation
predictions = np.array(predictions)

# Reshape predictions to treat each prediction as an independent example
# Assuming we want to classify each prediction into one of 3 classes
predictions_reshaped = predictions.reshape(-1, 1)  # Each row is a separate prediction

# Apply softmax to each prediction independently
# Here we're assuming each prediction needs to be evaluated against a set of classes (e.g., 1, 2, 3)
# So we'll expand the predictions to consider them against possible classes
expanded_predictions = np.tile(predictions_reshaped, (1, 3))  # Repeat each prediction 3 times for 3 classes

# Now apply softmax to each row (each prediction considered against three classes)
softmax_predictions = np.exp(expanded_predictions) / np.sum(np.exp(expanded_predictions), axis=1, keepdims=True)

# Convert softmax outputs to class labels (assuming classes start from 1)
final_predictions = np.argmax(softmax_predictions, axis=1) + 1  #

print("Softmax Predictions:", softmax_predictions)
print("Final Predictions:", final_predictions)


clamped_predictions = [max(1, min(3, x)) for x in predictions]
rounded_predictions = [round(x) for x in clamped_predictions]
print("Clamped and rounded Predictions:", rounded_predictions)