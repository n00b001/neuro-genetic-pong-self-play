from ga import load_best_population
from human_controls import HumanPlayer1
from main import perform_episode
from numpy_nn import create_model_from_genes
from my_intense_pong import MyPong


def main():
    population = load_best_population()
    individual = population[0]
    run_against_human(individual)


def run_against_human(individual=None):
    right_model = create_model_from_genes(individual)
    left_model = HumanPlayer1()

    # env = retro.make('Pong-Atari2600', state='Start.2P', players=2)
    #
    # env.use_restricted_actions = retro.Actions.FILTERED
    # env.reset()
    env = MyPong(
        left_model,
        right_model
    )
    env.reset()
    perform_episode(env, left_model, right_model, True, 1)


if __name__ == '__main__':
    main()
