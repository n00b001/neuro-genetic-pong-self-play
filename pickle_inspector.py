from ga import load_best_population
from main import evaluate


def main():
    population = load_best_population()
    individual = population[0]
    evaluate(individual=individual, render=True)


if __name__ == '__main__':
    main()
