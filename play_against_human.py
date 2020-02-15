from ga import load_best_population
from main import run_against_human


def main():
    population, hall_of_fame = load_best_population()
    individual = population[0]
    run_against_human(individual)


if __name__ == '__main__':
    main()
