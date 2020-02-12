from ga import load_best_population
from main import evaluate


def main():
    population, hall_of_fame = load_best_population()
    individual = population[0]
    evaluate(individual=individual, hall_of_fame=hall_of_fame, render=True)


if __name__ == '__main__':
    main()
