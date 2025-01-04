import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

def imputation(data, filename):
    imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=50, random_state=0)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    fileDirectory = f"output/Imputed_{filename}_using_MICE.csv"
    imputed_data.to_csv(fileDirectory, index=False)
    return fileDirectory