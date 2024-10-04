import random
import numpy as np

from seismic_fx import stalta_adaptive
from signal_fx import sliding_window_rms

from deap import base, creator, tools, algorithms
import scipy.signal

# Create fitness and individual types in DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define a simple toolbox using DEAP
toolbox = base.Toolbox()

# Parameter bounds for STA, LTA, and threshold
STA_MIN, STA_MAX = 1, 10   # STA window range in seconds
LTA_MIN, LTA_MAX = 5, 50   # LTA window range in seconds
THRESH_MIN, THRESH_MAX = 1, 5  # Threshold range
RMS_WINDOW_MIN, RMS_WINDOW_MAX = 1, 5
RMS_SCALE_MIN, RMS_SCALE_MAX = 1, 5

# Random initialization functions for STA, LTA, Threshold, RMS window, and RMS scale
toolbox.register("attr_sta", random.uniform, STA_MIN, STA_MAX)
toolbox.register("attr_lta", random.uniform, LTA_MIN, LTA_MAX)
toolbox.register("attr_threshold", random.uniform, THRESH_MIN, THRESH_MAX)
toolbox.register("rms_window", random.uniform, RMS_WINDOW_MIN, RMS_WINDOW_MAX)
toolbox.register("rms_scale", random.uniform, RMS_SCALE_MIN, RMS_SCALE_MAX)

# Create an individual (STA, LTA, Threshold, RMS Window, RMS Scale)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_sta, toolbox.attr_lta, toolbox.attr_threshold,
                  toolbox.rms_window, toolbox.rms_scale), n=1)

# Create a population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compute STA/LTA ratio for event detection using RMS scaling
def compute_sta_lta(signal, sta_window, lta_window, threshold, rms_window, rms_scale, sampling_rate=100):
    """
    Compute the STA/LTA detection method, where the STA and LTA windows are scaled by the RMS of the signal.
    
    Parameters:
    signal : array-like
        Seismic signal.
    sta_window : float
        Short-Term Average window (in seconds).
    lta_window : float
        Long-Term Average window (in seconds).
    threshold : float
        Detection threshold.
    rms_window : float
        Window length for RMS calculation (in seconds).
    rms_scale : float
        Scaling factor for STA/LTA windows based on RMS.
    sampling_rate : int
        Sampling rate of the signal (samples per second).
    
    Returns:
    detections : list
        List of detected event indices.
    """
    
    # Compute the RMS signal over the specified rms_window
    rms_signal = sliding_window_rms(signal, rms_window * sampling_rate)
    
    # Scale STA/LTA windows using the RMS value
    scaled_sta_window = sta_window * rms_scale * rms_signal
    scaled_lta_window = lta_window * rms_scale * rms_signal
    
    # Compute STA/LTA ratio using the scaled windows
    sta_lta_ratio, _, _ = stalta_adaptive(None, signal, sampling_rate, 
                                         sta_factor=scaled_sta_window, 
                                         lta_factor=scaled_lta_window)

    # Detect events where STA/LTA exceeds the threshold
    detections = np.where(sta_lta_ratio > threshold)[0]
    
    return detections

# Fitness function: minimize the distance between detections and nearest ground truth
def fitness_function(individual):
    sta_window, lta_window, threshold, rms_window, rms_scale = individual
    sampling_rate = 100  # Example sampling rate, modify as needed
    
    total_distance = 0
    penalty = 0

    # Loop over all signals and corresponding annotations
    for signal, annotations in zip(signals_array, annotations_array):
        # Get detections based on the current individual parameters
        detections = compute_sta_lta(signal, sta_window, lta_window, threshold, rms_window, rms_scale, sampling_rate=sampling_rate)
        
        # Penalize if there are too many detections (e.g., more than 10)
        if len(detections) > 10:
            penalty += 1000000 * (len(detections) - 10)  # Large penalty if more than 10 detections
            continue  # Skip to the next signal if penalty applies

        # Penalize if there are fewer than 1 detection
        if len(detections) < 1:
            penalty += 1000000  # Large penalty if fewer than 1 detection
            continue  # Skip to the next signal if penalty applies

        distances = []
        for detection in detections:
            nearest_annotation = min(annotations, key=lambda x: abs(x - detection))
            distances.append(abs(detection - nearest_annotation))

        total_distance += sum(distances)

    # Return the total distance and the accumulated penalty for all signals
    return total_distance + penalty,

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
    # Example usage: signals_array and annotations_array need to be passed to main
    # signals_array = [signal1, signal2, signal3, ...]
    # annotations_array = [annotations1, annotations2, annotations3, ...]
    main()
