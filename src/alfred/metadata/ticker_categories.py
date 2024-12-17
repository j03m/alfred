import json


class TickerCategories:
    def __init__(self, file_name):
        # Load JSON data from the provided file
        self.file_name = file_name
        with open(file_name, 'r') as file:
            self.data = json.load(file)
        self.version = 0

    # todo kill all the bad tickers
    def purge(self, bad_tickers):
        if len(bad_tickers) > 0:
            print("TICKER PURGE OF: ", bad_tickers)
            for category, tickers in self.data.items():
                # Use a list comprehension to efficiently filter out bad tickers
                self.data[category] = [ticker for ticker in tickers if ticker not in bad_tickers]
            self.version += 1
    def save(self):
        if self.version != 0:
            file_name = self.file_name + f".{self.version}.json"
            print("Versioning ticker file: ", file_name)
        else:
            file_name = self.file_name
        with open(file_name, 'w') as file:
            json.dump(self.data, file)

    def get(self, categories):
        # Initialize an empty list to store the combined results
        combined_list = []

        # Loop through each category and extend the combined_list
        for category in categories:
            if category in self.data:
                combined_list.extend(self.data[category])

        return combined_list
