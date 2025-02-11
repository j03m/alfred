import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Strength of Correlation:
0.00 to 0.19: Very weak
0.20 to 0.39: Weak
0.40 to 0.59: Moderate
0.60 to 0.79: Strong
0.80 to 1.00: Very strong
'''

df = pd.read_csv('results/training-pm-data.csv')

df = df.drop(['Date', 'ID'], axis=1)

correlation_with_label = df.corrwith(df['Rank'])

# Sort correlations to see highest first
correlation_with_label = correlation_with_label.sort_values(ascending=False)
for feature, correlation in correlation_with_label.items():
    print(f"{feature:<25} {correlation:.6f}")