import numpy as np
import math

def ESC(N, max_iter, lb, ub, dim, fobj):
    # Ensure bounds are arrays
    if np.isscalar(lb):
        lb = np.ones(dim) * lb
        ub = np.ones(dim) * ub

    # Initialize population
    population = np.random.rand(N, dim) * (ub - lb) + lb
    fitness = np.array([fobj(ind) for ind in population])
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

        # Boundary control
        new_population = np.clip(new_population, lb, ub)

        # Evaluate new population
        new_fitness = np.array([fobj(ind) for ind in new_population])

        # Replace individuals with better fitness
        for i in range(N):
            if new_fitness[i] < fitness[i]:
                population[i, :] = new_population[i, :]
                fitness[i] = new_fitness[i]

        # Update elite pool
        idx = np.argsort(fitness)
        fitness = fitness[idx]
        population = population[idx]
        best_solutions = population[:elite_size, :]

        # Record best fitness
        fitness_history.append(fitness[0])
        t += 1

    return fitness[0], population[0, :], fitness_history

def test_function(x):
    # this is a simple test function
    return sum(x**2)

#parameter set
N = 50
Max_iter = 1000
lb = -10
ub = 10
dim = 10

Best_score,Best_pos, fitness_history = ESC(N, Max_iter, lb, ub, dim, test_function)

print("Best Position：", Best_pos)
print("Best Score：", Best_score)