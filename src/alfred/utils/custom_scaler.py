import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import joblib
import re

SUPPORTED_AUGMENTS = ["log"]

class CustomScaler:
    def __init__(self, config, df):
        self.config = config
        self.scaler_mapping = {}
        self.augment_mapping = {}
        self.scalers = {}
        self._process_config(df)

    def _process_config(self, df):
        for entry in self.config:
            scaler_type = entry.get('type', 'standard')
            augment_type = entry.get('augment', None)
            columns = entry.get('columns', [])
            regex_pattern = entry.get('regex', None)

            if regex_pattern:
                pattern = re.compile(regex_pattern)
                matched_columns = [col for col in df.columns if pattern.match(col)]
                columns.extend(matched_columns)

            for column in columns:
                if scaler_type == "standard":
                    self.scaler_mapping[column] = StandardScaler()
                elif scaler_type == "minmax":
                    self.scaler_mapping[column] = MinMaxScaler()
                elif scaler_type == "robust":
                    self.scaler_mapping[column] = RobustScaler()
                else:
                    raise ValueError(f"Unsupported scaler type: {scaler_type}")

                if augment_type and augment_type not in SUPPORTED_AUGMENTS:
                    raise ValueError(f"Unsupported augment type: {augment_type}")

                self.augment_mapping[column] = augment_type

    def fit(self, df):

        for column, scaler in self.scaler_mapping.items():
            augment = self.augment_mapping.get(column, None)
            if augment == 'log':
                temp_col= np.log1p(df[[column]])
                scaler.fit(temp_col)
            else:
                scaler.fit(df[[column]])


    def transform(self, df, in_place = False):
        if not in_place:
            df = df.copy()

        for column, scaler in self.scaler_mapping.items():
            augment = self.augment_mapping.get(column, None)
            if augment == 'log':
                temp_col = np.log1p(df[[column]])
                df[column] = scaler.transform(temp_col)
            else:
                df[column] = scaler.transform(df[[column]])
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

# # Example usage
# config = [
#     {'columns': ['a', 'b', 'c'], 'type': 'standard', 'augment': None},
#     {'columns': ['d', 'e'], 'type': 'minmax', 'augment': 'log'},
#     {'regex': r'^f.*', 'type': 'standard'}
# ]
#
# # Assuming df is your DataFrame
# scaler = CustomScaler(config)
# scaled_df = scaler.fit_transform(df)
# scaler.serialize('scalers.pkl')
