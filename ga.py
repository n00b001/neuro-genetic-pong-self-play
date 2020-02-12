import glob
import os
import pickle
import random

from deap import base, tools, creator
from scoop import futures

from config import *
from utils import calculate_gene_size


def load_or_create_pop():
    global hall_of_fame, population

    population = load_latest_population()

    loaded_pop_size = 0
    if population is not None:
        loaded_pop_size = len(population)

    if loaded_pop_size >= POPULATION_SIZE:
        return population[:POPULATION_SIZE]

    _population = toolbox.population(n=(POPULATION_SIZE - loaded_pop_size))

    if loaded_pop_size == 0:
        return _population
    return population + _population


def load_latest_population():
    list_of_files = glob.glob('checkpoints/checkpoints/*')
    if len(list_of_files) > 0:
        checkpoint = max(list_of_files, key=os.path.getctime)
        return load_population_from_file(checkpoint)
    else:
        return None


def load_population_from_file(checkpoint):
    global hall_of_fame, NETWORK_SHAPE
    print("Loading: {}".format(checkpoint))
    with open(checkpoint, "rb") as cp_file:
        cp = pickle.load(cp_file)
    _population = cp["population"]
    _sorted_population = sorted(_population, key=lambda x: x.fitness.values[0], reverse=True)
    random.setstate(cp["rndstate"])
    if "hall_of_fame" in cp:
        hall_of_fame = cp["hall_of_fame"]
    if "network_shape" in cp:
        NETWORK_SHAPE = cp["network_shape"]
    return _sorted_population, hall_of_fame


def load_best_population():
    global population, hall_of_fame
    list_of_files = glob.glob('checkpoints/*')
    best_score = 0
    best_checkpoint = None
    for f in list_of_files:
        try:
            population, hall_of_fame = load_population_from_file(f)
            if best_score < population[0].fitness.values[0]:
                best_score = population[0].fitness.values[0]
                best_checkpoint = f
        except Exception as e:
            print(e)
    if best_checkpoint is not None:
        print("Loading best model: {}\nwith score: {}".format(best_checkpoint, best_score))
        # We have to load "twice" because we need to load the hall of fame/NN structure
        return load_population_from_file(best_checkpoint)
    else:
        return None


toolbox = base.Toolbox()
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_AMOUNT)

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox.register("map", futures.map)

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=calculate_gene_size())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=CROSSOVER_BLEND_ALPHA)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=GAUSSIAN_MUTATION_MEAN, sigma=GAUSSIAN_MUTATION_SIGMA,
                 indpb=PROBABILITY_OF_MUTATING_A_SINGLE_GENE)
# toolbox.register("select", tools.selBest, TOURN_SIZE)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

population = load_or_create_pop()
# this has to be called again because the gene size might have changed after loading the checkpoint
# and we can't do that first because the "individual" class hasn't been instantiated
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=calculate_gene_size())
