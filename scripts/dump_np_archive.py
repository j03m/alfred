#!/usr/bin/env python3
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="File to use", default="./models/ppo_mlp_policy_simple_env/evaluations.npz")
args = parser.parse_args()

# Load the .npz file
data = np.load(args.file)

# Convert the arrays to lists and write to a JSON file
data_dict = {key: data[key].tolist() for key in data.files}


print(json.dumps(data_dict))
