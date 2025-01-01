import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from deap import base, creator, tools, algorithms
import split_data
from pandasgui import show

def define_bounds_and_indices(missed_data, full_data):
    bounds_list = []
    col_to_indices = {}
    solution_start = 0

    for col in missed_data.columns:
        if missed_data[col].isna().any():
            col_min, col_max = full_data[col].min(), full_data[col].max()
            num_missing = missed_data[col].isna().sum()
            col_to_indices[col] = (solution_start, solution_start + num_missing)
            solution_start += num_missing
            bounds_list.extend([(col_min, col_max)] * num_missing)
    
    return bounds_list, col_to_indices

def fitness_function(individual, missed_data, col_to_indices, X_train, y_train, X_test, y_test):
    tmp_missed_data = missed_data.copy()
    for col, (start, end) in col_to_indices.items():
        tmp_missed_data.loc[tmp_missed_data[col].isna(), col] = individual[start:end]
    
    X_nan_test = tmp_missed_data.iloc[:, :-1]
    Y_nan_test = tmp_missed_data.iloc[:, -1]
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

def apply_optimized_solution(missed_data, best_ind, col_to_indices, data):
    for col, (start, end) in col_to_indices.items():
        col_min, col_max = data[col].min(), data[col].max()
        missed_data.loc[missed_data[col].isna(), col] = np.clip(best_ind[start:end], col_min, col_max)

    return missed_data

def imputation_data(data, filename, test_size):
    missed_data = split_data.missing_data()
    
    bounds_list, col_to_indices = define_bounds_and_indices(missed_data, data)
    
    X_train, y_train, X_test, y_test = split_data.split_and_clean_data(test_size)
    
    toolbox = setup_genetic_algorithm(bounds_list)
    toolbox.register("evaluate", fitness_function, missed_data=missed_data, col_to_indices=col_to_indices, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    best_ind = run_genetic_algorithm(toolbox)
    
    imputed_data = apply_optimized_solution(missed_data, best_ind, col_to_indices, data)
    fileDirectory = f"output/Imputed_{filename}_using_GA.csv"
    imputed_data.to_csv(fileDirectory, index=False)
    return fileDirectory
    
def imputation_std_data(data, filename, test_size):
    missed_data = split_data.missing_data()
    
    bounds_list, col_to_indices = define_bounds_and_indices(missed_data, data)
    
    X_train, y_train, X_test, y_test = split_data.split_and_clean_std_data(test_size)
    
    toolbox = setup_genetic_algorithm(bounds_list)
    toolbox.register("evaluate", fitness_function, missed_data=missed_data, col_to_indices=col_to_indices, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    best_ind = run_genetic_algorithm(toolbox)
    
    imputed_data = apply_optimized_solution(missed_data, best_ind, col_to_indices, data)
    fileDirectory = f"output/Imputed_{filename}_using_GA.csv"
    imputed_data.to_csv(fileDirectory, index=False)
    return fileDirectory