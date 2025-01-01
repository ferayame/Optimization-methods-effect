import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms

def test_with_and_without_standardization(X_train, X_test, y_train, y_test, classifier, title):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics, conf_mat = evaluate_performance(y_test, y_pred)
    print("Before Standardization:")
    visualize_performance(metrics, conf_mat, title)

    classifier.fit(X_train_std, y_train)
    y_pred_std = classifier.predict(X_test_std)
    metrics_std, conf_mat_std = evaluate_performance(y_test, y_pred_std)
    print("After Standardization:")
    visualize_performance(metrics_std, conf_mat_std, title)
    
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
        ax_pie.pie(sizes, labels=pie_labels, colors=pie_colors, autopct='%1.2f%%', startangle=140)
        ax_pie.set_title(f'{metric} Pie Chart',fontsize=12, fontweight='bold')

    
    # descriptions
    fig.text(0.5, 0.02, 'This figure visualizes performance metrics, confusion matrix, pie charts, and a histogram.', ha='center', fontsize=13,fontweight='bold')
    fig.text(0.5, 0.98, f'Displaying Results - {title}', ha='center', fontsize=16, fontweight='bold')

    #layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    

def evaluate_performance(y_true, y_pred):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred, average='macro'),
        'Precision': precision_score(y_true, y_pred, average='macro'),
        'F1 Score': f1_score(y_true, y_pred, average='macro'),
    }
    conf_mat = confusion_matrix(y_true, y_pred)
    return metrics, conf_mat










def define_bounds_and_indices(df_nan, data):
    bounds_list = []
    col_to_indices = {}
    solution_start = 0

    for col in df_nan.columns:
        if df_nan[col].isna().any():
            col_min, col_max = data[col].min(), data[col].max()
            num_missing = df_nan[col].isna().sum()
            col_to_indices[col] = (solution_start, solution_start + num_missing)
            solution_start += num_missing
            bounds_list.extend([(col_min, col_max)] * num_missing)
    
    return bounds_list, col_to_indices


def fitness_function(individual, df_nan, col_to_indices, X_train, y_train, X_test, y_test):
    df_temp = df_nan.copy()
    for col, (start, end) in col_to_indices.items():
        df_temp.loc[df_temp[col].isna(), col] = individual[start:end]
    
    X_nan_test = df_temp.iloc[:, :-1]
    Y_nan_test = df_temp.iloc[:, -1]
    X_test_combined = pd.concat([X_test, X_nan_test], axis=0)
    Y_test_combined = pd.concat([y_test, Y_nan_test], axis=0)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_combined = knn.predict(X_test_combined)
    accuracy = accuracy_score(Y_test_combined, y_pred_combined)
    
    return accuracy,

def setup_genetic_algorithm(bounds_list):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: random.uniform(bounds_list[0][0], bounds_list[0][1]))
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [random.uniform(bounds[0], bounds[1]) for bounds in bounds_list])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def run_genetic_algorithm(toolbox, NGEN=40, CXPB=0.7, MUTPB=0.2):
    population = toolbox.population(n=50)

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
    
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
    
        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, k=1)[0]
    return best_ind

def apply_optimized_solution(df_nan, best_ind, col_to_indices, data):
    for col, (start, end) in col_to_indices.items():
        col_min, col_max = data[col].min(), data[col].max()
        df_nan.loc[df_nan[col].isna(), col] = np.clip(best_ind[start:end], col_min, col_max)

    return df_nan

data = load_data('water_potability.csv')
detect_missing_values(data)
    
df_no_nan = data.dropna()
detect_missing_values(df_no_nan)

X = df_no_nan.iloc[:, :-1]
Y = df_no_nan.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
test_with_and_without_standardization(X_train, X_test, y_train, y_test, KNeighborsClassifier(n_neighbors=5), 'KNN Classifier')
test_with_and_without_standardization(X_train, X_test, y_train, y_test, MLPClassifier(random_state=42, max_iter=1000), 'MLP Classifier')

df_nan = data[data.isna().any(axis=1)].copy()

missing_rows, missing_cols = np.where(df_nan.isna())

bounds_list, col_to_indices = define_bounds_and_indices(df_nan, data)
    
toolbox = setup_genetic_algorithm(bounds_list)
toolbox.register("evaluate", fitness_function, df_nan=df_nan, col_to_indices=col_to_indices, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
best_ind = run_genetic_algorithm(toolbox)
print(f"Best individual is {best_ind}, with fitness: {best_ind.fitness.values[0]}")
    
df_nan_imputed = apply_optimized_solution(df_nan, best_ind, col_to_indices, data)
X = df_nan_imputed.iloc[:, :-1]
Y = df_nan_imputed.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
test_with_and_without_standardization(X_train, X_test, y_train, y_test, KNeighborsClassifier(n_neighbors=5), 'KNN Classifier')
test_with_and_without_standardization(X_train, X_test, y_train, y_test, MLPClassifier(random_state=42, max_iter=1000), 'MLP Classifier')

final_dataset = pd.concat([df_no_nan, df_nan_imputed]).sort_index()
    
print("Final Dataset:\n", final_dataset)
print(final_dataset.describe())
print("Original Data Stats (Before NaN):")
print(data.describe())
print("Missing Values Check:")
print(final_dataset.isna().sum())

# final_dataset.to_csv('final_imputed_data_GA.csv', index=False) 
# print("Final imputed dataset saved to 'final_imputed_data.csv'")

data_full_nan = data.copy()
bounds_list, col_to_indices = define_bounds_and_indices(data_full_nan, data)
    
toolbox = setup_genetic_algorithm(bounds_list)
toolbox.register("evaluate", fitness_function, df_nan=data_full_nan, col_to_indices=col_to_indices, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
best_ind = run_genetic_algorithm(toolbox)
print(f"Best individual is {best_ind}, with fitness: {best_ind.fitness.values[0]}")
    
data_full_nan_imputed = apply_optimized_solution(data_full_nan, best_ind, col_to_indices, data)
X = data_full_nan_imputed.iloc[:, :-1]
Y = data_full_nan_imputed.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
test_with_and_without_standardization(X_train, X_test, y_train, y_test, KNeighborsClassifier(n_neighbors=5), 'KNN Classifier')
test_with_and_without_standardization(X_train, X_test, y_train, y_test, MLPClassifier(random_state=42, max_iter=1000), 'MLP Classifier')

final_dataset = data_full_nan_imputed
    
print("Final Dataset:\n", final_dataset)
print(final_dataset.describe())
print("Original Data Stats (Before NaN):")
print(data.describe())
print("Missing Values Check:")
print(final_dataset.isna().sum())

# Save
# final_dataset.to_csv('final_imputed_data_GA_2.csv', index=False) 
# print("Final imputed dataset saved to 'final_imputed_data.csv'")

