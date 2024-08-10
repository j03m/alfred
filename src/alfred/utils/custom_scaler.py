import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import joblib

'''
Apply custom column scaling and saving of said scalars to a dataframe
I only implemented minmax and standard for now but arguably open to addition later

# Example usage
config = [
    {'columns': ['a', 'b', 'c'], 'type': 'standard', 'augment': None},
    {'columns': ['d', 'e'], 'type': 'minmax', 'augment': 'log'},
    {'columns': ['f'], 'type': 'standard'}
]

# Assuming df is your DataFrame
scaler = CustomScaler(config)
scaled_df = scaler.fit_transform(df)
scaler.serialize('scalers.pkl')

# To load later:
scaler = CustomScaler.load('scalers.pkl')
scaled_df = scaler.transform(df)
'''

SUPPORTED_AUGMENTS = ["log"]

class CustomScaler:
    def __init__(self, config):
        self.config = config
        self.scaler_mapping = {}
        self.augment_mapping = {}
        self.scalers = {}
        self._process_config()

    def _process_config(self):
        for entry in self.config:
            scaler_type = entry.get('type', 'standard')
            augment_type = entry.get('augment', None)

            for column in entry['columns']:
                if scaler_type == "standard":
                    self.scaler_mapping[column] = StandardScaler()
                elif scaler_type == "minmax":
                    self.scaler_mapping[column] = MinMaxScaler()
                else:
                    raise ValueError(f"Unsupported scaler type: {scaler_type}")

                if augment_type not in SUPPORTED_AUGMENTS:
                    raise ValueError(f"Unsupported augment type: {augment_type}")

                self.augment_mapping[column] = augment_type

    def _apply_augmentation(self, df):
        for column, augment in self.augment_mapping.items():
            if augment == 'log':
                df[column] = np.log1p(df[column])
        return df

    def fit(self, df):
        df = self._apply_augmentation(df)
        for column, scaler in self.scaler_mapping.items():
            scaler.fit(df[[column]])

    def transform(self, df):
        df = self._apply_augmentation(df)
        for column, scaler in self.scaler_mapping.items():
            df[[column]] = scaler.transform(df[[column]])
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_scaler(self, column_name):
        if column_name in self.scaler_mapping:
            return self.scaler_mapping[column_name]
        else:
            raise ValueError(f"No scaler found for column: {column_name}")

    def inverse_transform(self, df):
        for column, scaler in self.scaler_mapping.items():
            df[[column]] = scaler.inverse_transform(df[[column]])
            if self.augment_mapping[column] == 'log':
                df[column] = np.expm1(df[column])  # Reverse the log1p operation
        return df

    def serialize(self, path):
        # Save the actual scalers along with the mappings
        joblib.dump({
            "config": self.config,
            "scaler_mapping": self.scaler_mapping,
            "augment_mapping": self.augment_mapping,
            "scalers": {col: scaler for col, scaler in self.scaler_mapping.items()}
        }, path)

    @staticmethod
    def load(path):
        data = joblib.load(path)
        custom_scaler = CustomScaler(data['config'])
        custom_scaler.scaler_mapping = data['scalers']
        custom_scaler.augment_mapping = data['augment_mapping']
        return custom_scaler

    def get_scaled(self, df):
        return self.transform(df.copy())


