import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('data/water_potability.csv')
std_data = pd.read_csv('data/standardized_water_potability.csv')

def split(data, test_size):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def _split_with_nan(data, test_size=0.2):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train_initial, X_test_initial, y_train_initial, y_test_initial = train_test_split(X, y, test_size=test_size, random_state=42)
    
    X_train = X_train_initial.dropna()
    y_train = y_train_initial[X_train.index]
    
    return X_train, y_train, X_train_initial, X_test_initial, y_train_initial, y_test_initial

def split_and_clean_std_data(test_size=0.2):
    X_train, y_train, _, X_test_initial, _, y_test_initial = _split_with_nan(std_data, test_size)
    X_test = X_test_initial.dropna()
    y_test = y_test_initial[X_test.index]
    return X_train, y_train, X_test, y_test
    
def split_and_clean_data(test_size=0.2):
    X_train, y_train, _, X_test_initial, _, y_test_initial = _split_with_nan(data, test_size)
    X_test = X_test_initial.dropna()
    y_test = y_test_initial[X_test.index]
    return X_train, y_train, X_test, y_test

def split_and_mix_std_data(test_size=0.2):
    X_train, y_train, X_train_initial, X_test_initial, y_train_initial, y_test_initial = _split_with_nan(std_data, test_size)
    X_test = pd.concat([X_test_initial, X_train_initial.loc[X_train_initial.isnull().any(axis=1)]])
    y_test = pd.concat([y_test_initial, y_train_initial.loc[X_train_initial.isnull().any(axis=1)]])
    return X_train, y_train, X_test, y_test

def split_and_mix_data(test_size=0.2):
    X_train, y_train, X_train_initial, X_test_initial, y_train_initial, y_test_initial = _split_with_nan(data, test_size)
    X_test = pd.concat([X_test_initial, X_train_initial.loc[X_train_initial.isnull().any(axis=1)]])
    y_test = pd.concat([y_test_initial, y_train_initial.loc[X_train_initial.isnull().any(axis=1)]])
    return X_train, y_train, X_test, y_test

def missing_data():
    return data[data.isna().any(axis=1)].copy()