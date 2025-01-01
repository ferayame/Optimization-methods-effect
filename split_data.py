import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('webapp/data/water_potability.csv')
std_data = pd.read_csv('webapp/data/standardized_water_potability.csv')

def split(data, test_size):
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test