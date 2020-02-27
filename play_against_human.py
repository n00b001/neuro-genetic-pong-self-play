from dumb_ais import RandomHardcodedAi
from ga import load_latest_population
from human_controls import HumanPlayer1
from main import perform_episode
from my_intense_pong import MyPong
from numpy_nn import create_model_from_genes


def main():
    population = load_latest_population()
    if population is None or len(population) == 0:
        individual = None
    else:
        individual = population[0]
    run_against_human(individual)


def run_against_human(individual=None):
    left_model = RandomHardcodedAi(1.0)
    right_model = HumanPlayer1()

    if individual is not None:
        print("Loading enemy from genes...")
        left_model = create_model_from_genes(individual)
        right_model = create_model_from_genes(individual)
        # right_model = RandomHardcodedAi(1.0)

    env = MyPong(
        render=True
    )
    env.reset()
    perform_episode(env, left_model, right_model, True, 1)


if __name__ == '__main__':
    main()
