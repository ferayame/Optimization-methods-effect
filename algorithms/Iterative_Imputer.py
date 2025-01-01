import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score ,recall_score,precision_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import IterativeImputer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def detect_missing_values(data):
    for i in data.columns:
        if data[i].isna().sum() > 0:
            print(f'{i}',data[i].isna().sum())
            
def imputation_with_iterative_imputer(data):
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, columns=data.columns)

def evaluate_performance(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred, average='macro'),
        'Precision': precision_score(y_true, y_pred, average='macro'),
        'F1 Score': f1_score(y_true, y_pred, average='macro'),
    }
    conf_mat = confusion_matrix(y_true, y_pred)
    return metrics, conf_mat

def visualize_performance(metrics, conf_mat, title):
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig = plt.figure(figsize=(20,10))
    # Bar plot 
    ax1 = fig.add_subplot(2, 3, 1)
    sns.barplot(ax=ax1, x=metric_names, y=metric_values,hue=metric_names, palette='viridis')
    ax1.set_ylim(0, 1)
    ax1.set_title(f'{title} - Performance Metrics',fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xlabel('Metrics', fontsize=12)
    for i, v in enumerate(metric_values):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', color='black', fontweight='bold')

    # Confusion matrix heatmap
    ax2 = fig.add_subplot(2, 3, 2)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Potable', 'Potable'],
                yticklabels=['Not Potable', 'Potable'], ax=ax2)
    ax2.set_title(f'{title} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)

    # Pie charts 
    pie_colors = ['#4CAF50', '#FF5252']
    pie_labels = ['Correct', 'Incorrect']
    for i, (metric, value) in enumerate(metrics.items()):
        ax_pie = fig.add_subplot(2, 6, 7 + i)
        sizes = [value, 1 - value]
        ax_pie.pie(sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=140)
        ax_pie.set_title(f'{metric} Pie Chart',fontsize=12, fontweight='bold')

    
    # descriptions
    fig.text(0.5, 0.02, 'This figure visualizes performance metrics, confusion matrix, pie charts, and a histogram.', ha='center', fontsize=13,fontweight='bold')
    fig.text(0.5, 0.98, f'Displaying Results - {title}', ha='center', fontsize=16, fontweight='bold')

    #layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
def test_with_and_without_standardization(X_train, X_test, y_train, y_test, classifier, title):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics, conf_mat = evaluate_performance(y_test, y_pred)
    
    classifier.fit(X_train_std, y_train)
    y_pred_std = classifier.predict(X_test_std)
    metrics_std, conf_mat_std = evaluate_performance(y_test, y_pred_std)

    return print("Befor Standardization \n"), visualize_performance(metrics, conf_mat, title), print("After Standardization \n"),visualize_performance(metrics_std, conf_mat_std, title)

data = load_data("water_potability.csv")
detect_missing_values(data)

df_no_nan = data.dropna()  #data without Nan values
detect_missing_values(df_no_nan)

X = df_no_nan.iloc[:, :-1]  
Y = df_no_nan.iloc[:,-1]  

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
test_with_and_without_standardization(X_train, X_test, y_train, y_test, KNeighborsClassifier(n_neighbors=5), "KNN Classifier")
test_with_and_without_standardization(X_train, X_test, y_train, y_test, MLPClassifier(random_state=42, max_iter=1000), "MLP Classifier")


df_nan = data[data.isna().any(axis=1)].copy() #data with Nan values
detect_missing_values(df_nan)

data_nan_imputed= imputation_with_iterative_imputer(df_nan)
detect_missing_values(data_nan_imputed)

X_nan_test = data_nan_imputed.iloc[:, :-1]
Y_nan_test = data_nan_imputed.iloc[:, -1]

X_test_combined = pd.concat([X_test, X_nan_test], axis=0)
Y_test_combined = pd.concat([y_test, Y_nan_test], axis=0)

test_with_and_without_standardization(X_train, X_test_combined, y_train, Y_test_combined, KNeighborsClassifier(n_neighbors=5), "KNN Classifier")
test_with_and_without_standardization(X_train, X_test_combined, y_train, Y_test_combined, MLPClassifier(random_state=42, max_iter=1000), "MLP Classifier")

print("The final data with all imputed values:\n")
final_data = pd.concat([df_no_nan, data_nan_imputed], axis=0)
print(final_data)
detect_missing_values(final_data)
print(final_data.describe())

#final_data.to_csv('final_data_imputed_with_IterImp.csv', index=False)
#final_data.to_excel('final_data_imputed_with_IterImp.xlsx', index=False)

full_data_imputed = imputation_with_iterative_imputer(data)
X = full_data_imputed.iloc[:, :-1]
Y = full_data_imputed.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
test_with_and_without_standardization(x_train, x_test, y_train, y_test, KNeighborsClassifier(n_neighbors=5), "KNN Classifier")
test_with_and_without_standardization(x_train, x_test, y_train, y_test, MLPClassifier(random_state=42, max_iter=1000), "MLP Classifier")
# Save 
#full_data_imputed.to_csv('full_data_imputed_with_IterImp_2.csv', index=False)
#full_data_imputed.to_excel('full_data_imputed_with_IterImp_2.xlsx', index=False)

