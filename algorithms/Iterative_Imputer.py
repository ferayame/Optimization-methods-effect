import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
          
def imputation(data, filename):
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    fileDirectory = f"output/Imputed_{filename}_using_Iterative_Imputer.csv"
    imputed_data.to_csv(fileDirectory, index=False)
    return fileDirectory
