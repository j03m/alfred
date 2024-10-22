import os
import re

def prune_old_versions(directory):
    # Updated regex pattern to match the device token dynamically (cpu, cuda, mps, etc.)
    # Matches the format lstm_30_128_1_0x216c6ee9_<device><number>.pth
    pattern = re.compile(r'(.*_)(cpu|cuda|mps)(\d+)\.pth')

    # Dictionary to store the latest version of each model based on the device token
    latest_versions = {}

    # Traverse the files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            match = pattern.match(filename)
            if match:
                model_name = match.group(1)  # The part before the device token
                device_token = match.group(2)  # The device token (cpu, cuda, mps)
                version = int(match.group(3))  # The version number

                # Create a unique key for each model+device combination
                model_device_key = f"{model_name}{device_token}"

                # Check if it's the latest version for this model+device combination
                if model_device_key not in latest_versions or version > latest_versions[model_device_key][1]:
                    latest_versions[model_device_key] = (filename, version)

    # Now we can delete the older versions
    for filename in os.listdir(directory):
        if filename.endswith('.pth'):
            match = pattern.match(filename)
            if match:
                model_name = match.group(1)
                device_token = match.group(2)
                version = int(match.group(3))

                # Create the key again to check if this is the latest version
                model_device_key = f"{model_name}{device_token}"

                # If this is not the latest version, delete the file
                if version != latest_versions[model_device_key][1]:
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

if __name__ == "__main__":
    # Specify the directory containing the model files
    model_directory = './models'  # Update this path if needed
    prune_old_versions(model_directory)
