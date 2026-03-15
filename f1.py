import random
import matplotlib.pyplot as plt

POP_SIZE = 10 
GENS = 40 
REPS = 10 
MUT_RANGE = 0.25 
MUT_PROB = 0.2 

def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        x = round(random.uniform(-5, 5),2)
        y = round(random.uniform(-5, 5),2)
        population.append([x, y])
    return population

def fitness_function(individual):
    x = individual[0]
    y = individual[1]
    result = round(x**2 + y**2, 2)
    return {"x": x, "y": y, "fitness": result}

def fps_selection(population_and_fitness):
    # 1. Calculate the sum of all fitness values
    total_fitness = sum(f['fitness'] for f in population_and_fitness)
    
    # 2. Pick a random threshold between 0 and the total sum
    pick = random.uniform(0, total_fitness)
    
    # 3. Iterate through the population and 'add up' fitness
    current_sum = 0
    for i in range(len(population_and_fitness)):
        current_sum += population_and_fitness[i]['fitness']
        
        # 4. As soon as the current_sum exceeds the pick, we found our winner
        if current_sum > pick:
            return population_and_fitness[i]

def rbs_selection(population_and_fitness):
    # 1. Sort dictionaries with fitness in ascending order
    # (Creating a new list, but the dictionaries inside are still the same objects)
    sorted_pop = sorted(population_and_fitness, key=lambda x: x['fitness'])
    
    n = len(sorted_pop)
    # 3. Calculate total_rank_sum = n * (n + 1) / 2
    total_rank_sum = n * (n + 1) / 2
    
    # 4. Pick a random threshold between 0 and total_rank_sum
    pick = random.uniform(0, total_rank_sum)
    
    # 5. Step through the sorted list
    current_sum = 0
    for i in range(n):
        # We use the index + 1 as the rank without adding it to the dictionary
        rank = i + 1
        current_sum += rank
        
        # 6. If current_sum > pick, we found our winner
        if current_sum > pick:
            return sorted_pop[i]

def binary_tournament_selection(population_and_fitness):
    pick1 = random.randint(0, len(population_and_fitness)-1)
    pick2 = random.randint(0, len(population_and_fitness)-1)
    contestant1 = population_and_fitness[pick1]['fitness']
    contestant2 = population_and_fitness[pick2]['fitness']
    if contestant1 > contestant2:
        return population_and_fitness[pick1]
    else:
        return population_and_fitness[pick2]

def truncation_selection(population_and_fitness):
    sorted_pop = sorted(population_and_fitness, key=lambda x: x['fitness'], reverse=True)
    return sorted_pop[:10]














# 1. Initialize population
pop = initialize_population()

# 2. Call fitness_function on every nested list and store results in a list of dictionaries
population_and_fitness = []
for individual in pop:
    fitness = fitness_function(individual)
    population_and_fitness.append(fitness)
    print(fitness)

# 3. Pass the list of dictionaries to selection functions
winner_fps = fps_selection(population_and_fitness)
print("\nWinner selected by FPS:", winner_fps)

winner_rbs = rbs_selection(population_and_fitness)
print("Winner selected by RBS:", winner_rbs)

winner_binary_tournament = binary_tournament_selection(population_and_fitness)
print("Winner selected by Binary Tournament:", winner_binary_tournament)

winner_truncation = truncation_selection(population_and_fitness)
print("Winner selected by Truncation:", winner_truncation)
