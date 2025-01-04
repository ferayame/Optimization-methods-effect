import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import split_data

def bounds(missed_data, full_data):
    upper_bound = []
    lower_bound = []
    solution_start = 0

    for col in missed_data.columns:
        if missed_data[col].isna().any():
            col_min, col_max = full_data[col].min(), full_data[col].max()
            num_missing = missed_data[col].isna().sum()
            solution_start += num_missing
            upper_bound.extend([col_max] * num_missing)
            lower_bound.extend([col_min] * num_missing)
    
    return upper_bound, lower_bound

def ESC(N, max_iter, lb, ub, dim, fobj, data, test_size):
    lb = np.array(lb)
    ub = np.array(ub)

    population = np.random.rand(N, dim) * (ub - lb) + lb
    fitness = np.array([fobj(data, ind, test_size) for ind in population])
    idx = np.argsort(fitness)
    fitness = fitness[idx]
    population = population[idx]

    elite_size = 5
    beta_base = 1.5
    t = 0
    mask_probability = 0.5
    fitness_history = []
    best_solutions = population[:elite_size, :]

    def adaptive_levy_weight(beta_base, dim, t, max_iter):
        beta = beta_base + 0.5 * np.sin(np.pi / 2 * t / max_iter)
        beta = np.clip(beta, 0.1, 2)
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        w = np.abs(u / np.abs(v) ** (1 / beta))
        return w / (np.max(w) + np.finfo(float).eps)

    while t < max_iter:
        panic_index = np.cos(np.pi / 2 * (t / (3 * max_iter)))
        idx = np.argsort(fitness)
        fitness = fitness[idx]
        population = population[idx]
        calm_count = int(round(0.15 * N))
        conform_count = int(round(0.35 * N))
        calm = population[:calm_count, :]
        conform = population[calm_count:calm_count + conform_count, :]
        panic = population[calm_count + conform_count:, :]
        calm_center = np.mean(calm, axis=0)

        new_population = population.copy()

        for i in range(N):
            if t / max_iter <= 0.5:
                if i < calm_count:
                    mask1 = np.random.rand(dim) > mask_probability
                    weight_vector1 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                    random_position = np.min(calm, axis=0) + np.random.rand(dim) * (
                                np.max(calm, axis=0) - np.min(calm, axis=0))
                    new_population[i, :] += mask1 * (weight_vector1 * (calm_center - population[i, :]) +
                                                     (random_position - population[i, :] + np.random.randn(
                                                         dim) / 50) )* panic_index
                elif i < calm_count + conform_count:
                    mask1 = np.random.rand(dim) > mask_probability
                    mask2 = np.random.rand(dim) > mask_probability
                    weight_vector1 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                    weight_vector2 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                    random_position = np.min(conform, axis=0) + np.random.rand(dim) * (
                                np.max(conform, axis=0) - np.min(conform, axis=0))
                    panic_individual = panic[np.random.randint(panic.shape[0])] if len(panic) > 0 else np.zeros(dim)
                    new_population[i, :] += mask1 * (weight_vector1 * (calm_center - population[i, :]) +
                                                     mask2 * weight_vector2 * (panic_individual - population[i, :]) +
                                                     (random_position - population[i, :] + np.random.randn(
                                                         dim) / 50) * panic_index)
                else:
                    mask1 = np.random.rand(dim) > mask_probability
                    mask2 = np.random.rand(dim) > mask_probability
                    weight_vector1 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                    weight_vector2 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                    elite = best_solutions[np.random.randint(elite_size)]
                    random_individual = population[np.random.randint(N)]
                    random_position = elite + weight_vector1 * (random_individual - elite)
                    new_population[i, :] += mask1 * (weight_vector1 * (elite - population[i, :]) +
                                                     mask2 * weight_vector2 * (random_individual - population[i, :]) +
                                                     (random_position - population[i, :] + np.random.randn(
                                                         dim) / 50) * panic_index)
            else:
                mask1 = np.random.rand(dim) > mask_probability
                mask2 = np.random.rand(dim) > mask_probability
                weight_vector1 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                weight_vector2 = adaptive_levy_weight(beta_base, dim, t, max_iter)
                elite = best_solutions[np.random.randint(elite_size)]
                random_individual = population[np.random.randint(N)]
                new_population[i, :] += mask1 * weight_vector1 * (elite - population[i, :]) + \
                                        mask2 * weight_vector2 * (random_individual - population[i, :])

        new_population = np.clip(new_population, lb, ub)

        new_fitness = np.array([fobj(data, ind, test_size) for ind in new_population])

        for i in range(N):
            if new_fitness[i] < fitness[i]:
                population[i, :] = new_population[i, :]
                fitness[i] = new_fitness[i]

        idx = np.argsort(fitness)
        fitness = fitness[idx]
        population = population[idx]
        best_solutions = population[:elite_size, :]

        # Record best fitness
        fitness_history.append(fitness[0])
        t += 1

    return fitness[0], population[0, :], fitness_history

def test_function(data, position, test_size):
    X_train, y_train, X_test, y_test = split_data.split_and_clean(data, test_size)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
        
    return -accuracy 

def impute_missing_values(data, best_position):
    dt = data.copy()
    pos_index = 0
    for col in dt.columns:
        if dt[col].isnull().any():
            num_missing = dt[col].isnull().sum()
            dt.loc[dt[col].isnull(), col] = best_position[pos_index:pos_index + num_missing]
            pos_index += num_missing
            
    return dt

def imputation(data, filename, test_size):
    missed_data = split_data.missing_data(data)
    
    lb, ub = bounds(missed_data, data)
    dim = len(ub)

    Best_score, Best_pos, fitness_history = ESC(N=50, max_iter=1000, lb=lb, ub=ub, dim=dim, fobj=test_function, data=data, test_size=test_size)
    imputed_data = impute_missing_values(data, Best_pos)
    fileDirectory = f"output/Imputed_{filename}_using_ESC.csv"
    imputed_data.to_csv(fileDirectory, index=False)
    return fileDirectory
