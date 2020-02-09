from main import load_population, evaluate


def main():
    population = load_population()

    individual = population[0]
    evaluate(individual=individual, render=True)


if __name__ == '__main__':
    main()
