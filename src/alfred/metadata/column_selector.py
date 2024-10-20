import json

class ColumnSelector:
    def __init__(self, file_name):
        # Load JSON data from the provided file
        with open(file_name, 'r') as file:
            self.data = json.load(file)

    def get(self, categories):
        # Initialize an empty list to store the combined results
        combined_list = []

        # Loop through each category and extend the combined_list
        for category in categories:
            if category in self.data:
                combined_list.extend(self.data[category["name"]])

        return combined_list
