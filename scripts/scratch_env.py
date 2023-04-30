#!/usr/bin/env python3
from machine_learning_finance import TraderEnv
import pandas as pd

df = pd.read_csv('./data/fake.csv')
env = TraderEnv("TEST", df, 2)
env.step(1)
env.step(2)
env.step(1)