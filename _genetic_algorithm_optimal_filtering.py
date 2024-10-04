# optimal filtering aims to define the optimal parameters like ``sta_window`` and ``lta_window`` of the filter,
# so that a predefined thresold will work, this needs the supervision from the output. 
# to avoid the class-imbalance problem, the method would be to minimize the distance between any detection and the 
# nearest annotation. This will be realized via the genetic algorithm to do exhausted search. 
# pip install deap
import random
import numpy as np

from seismic_fx import stalta_classic

from deap import base, creator, tools, algorithms
import scipy.signal

# Create fitness and individual types in DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define a simple toolbox using DEAP
toolbox = base.Toolbox()

# Parameter bounds for STA, LTA and threshold
STA_MIN, STA_MAX = 1, 10   # STA window range in seconds
LTA_MIN, LTA_MAX = 5, 50   # LTA window range in seconds
THRESH_MIN, THRESH_MAX = 1, 5  # Threshold range

# Random initialization functions for STA, LTA, Threshold
toolbox.register("attr_sta", random.uniform, STA_MIN, STA_MAX)
toolbox.register("attr_lta", random.uniform, LTA_MIN, LTA_MAX)
toolbox.register("attr_threshold", random.uniform, THRESH_MIN, THRESH_MAX)

# Create an individual (STA, LTA, Threshold)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_sta, toolbox.attr_lta, toolbox.attr_threshold), n=1)

# Create a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Seismic data (example) and ground truth annotations (for fitness calculation)
# Replace with your actual data
signal = np.random.randn(1000)  # Simulated seismic data
annotations = [200, 500, 800]   # Ground truth event locations (replace with actual data)

# Compute STA/LTA ratio for event detection
def compute_sta_lta(signal, sta_window, lta_window, threshold, sampling_rate=100):
    """
    Compute the STA/LTA detection method.
    
    Parameters:
    signal : array-like
        Seismic signal.
    sta_window : float
        Short-Term Average window (in seconds).
    lta_window : float
        Long-Term Average window (in seconds).
    threshold : float
        Detection threshold.
    sampling_rate : int
        Sampling rate of the signal (samples per second).
    
    Returns:
    detections : list
        List of detected event indices.
    """

    sta_lta_ratio, sta_out, lta_out = stalta_classic(None, signal, sampling_rate, 
                                                     sta_factor=sta_window, lta_factor=lta_window)

    # Detect events where STA/LTA exceeds the threshold
    detections = np.where(sta_lta_ratio > threshold)[0]
    
    return detections

# Fitness function: minimize the distance between detections and nearest ground truth
def fitness_function(individual):
    sta_window, lta_window, threshold = individual
    sampling_rate = 100  # Example sampling rate, modify as needed
    
    # Get detections based on the current individual parameters
    detections = compute_sta_lta(signal, sta_window, lta_window, threshold, sampling_rate)
    
    # Penalize if there are too many detections (more than 2)
    if len(detections) > 10:
        return 1000000 * (len(detections) - 10),  # Large penalty if more than 2 detections
    
    # Penalize if there are fewer than 2 detections
    if len(detections) < 1:
        return 1000000,  # Large penalty if less than 2 detections

    distances = []
    for detection in detections:
        nearest_annotation = min(annotations, key=lambda x: abs(x - detection))
        distances.append(abs(detection - nearest_annotation))
    
    total_distance = sum(distances)
    
    return total_distance,  # DEAP expects a tuple

def main():

    # Register the GA operations
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness_function)

    # Parallelize the GA using multiprocessing (optional)
    import multiprocessing
    toolbox.register("map", multiprocessing.Pool().map)

    # GA parameters
    population_size = 100
    num_generations = 50
    cx_prob = 0.5  # Probability of crossover
    mut_prob = 0.2  # Probability of mutation

    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Set up the statistics and Hall of Fame
    hof = tools.HallOfFame(1)  # Store the best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    
    # Run the Genetic Algorithm
    algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob,
                        ngen=num_generations, stats=stats, halloffame=hof, verbose=True)
    
    # Output the best individual
    print("Best individual (STA, LTA, Threshold):", hof[0])
    print("Best fitness score (total detection distance):", hof[0].fitness.values[0])

if __name__ == "__main__":
    main()
