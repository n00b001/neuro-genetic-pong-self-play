import datetime
import glob
import os
import pickle
import random
import time

import cv2
import numpy as np
import retro
from deap import base, algorithms
from deap import creator
from deap import tools
from scoop import futures

from config import MU, SIGMA, IND_PB, SCALE_FACTOR, GAME_TOP, GAME_BOTTOM, BALL_COLOUR, LEFT_GUY_COLOUR, \
    RIGHT_GUY_COLOUR, GENE_SIZE, SCALED_PADDLE_HEIGHT, GAMES_TO_PLAY, BLANK_ACTION, ALL_ACTIONS, RIGHT_ACTION_START, \
    RIGHT_ACTION_END, LEFT_ACTION_END, FPS, TIMEOUT_THRESH, POPULATION_SIZE, HALL_OF_FAME_AMOUNT, CX_PB, MUT_PB, N_GENS


def eval(individual):
    return evaluate(individual)


creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=IND_PB)
toolbox.register("select", tools.selBest)

toolbox.register("evaluate", eval)


def find_stuff(observation):
    small_img = cv2.resize(
        observation,
        (observation.shape[1] // SCALE_FACTOR, observation.shape[0] // SCALE_FACTOR),
        interpolation=cv2.INTER_NEAREST
    )
    chopped_observation = small_img[GAME_TOP // SCALE_FACTOR:GAME_BOTTOM // SCALE_FACTOR, :]
    ball_location = get_rect(chopped_observation, BALL_COLOUR)
    left_location = get_rect(chopped_observation, LEFT_GUY_COLOUR)
    right_location = get_rect(chopped_observation, RIGHT_GUY_COLOUR)
    return ball_location, left_location, right_location


def get_rect(chopped_observation, colour):
    indices = np.where(np.all(chopped_observation == colour, axis=-1))
    if len(indices[0]) == 0:
        return None
    return np.average(indices, axis=1)


def inferrance(ball_location, me, enemy, model):
    ret_val = [0, 0]
    if ball_location[0] < me[0]:
        ret_val[0] = 1
    elif ball_location[0] > me[0]:
        ret_val[1] = 1
    return ret_val


""":arg action[0] is up, action[1] is down"""


def keep_within_game_bounds_please(paddle, action):
    if paddle is not None:
        if paddle[0] < SCALED_PADDLE_HEIGHT:
            action = [0, 1]
        elif paddle[0] > (((GAME_BOTTOM - GAME_TOP) / SCALE_FACTOR) - SCALED_PADDLE_HEIGHT / 2):
            action = [1, 0]
    return action


def load_model_from_genes(individual):
    return None


def evaluate(individual=None, render=False):
    # env = retro.make('Pong-Atari2600', state='Start.2P', players=2)
    env = retro.make('Pong-Atari2600')
    env.use_restricted_actions = retro.Actions.FILTERED
    env.reset()

    model1 = None
    model2 = load_model_from_genes(individual)

    st = time.time()
    total_score = []
    for i in range(GAMES_TO_PLAY):
        score_info = perform_episode(env, model1, model2, render)
        total_score.append(score_info)

    total_score = np.array(total_score)
    left_total_score = sum(total_score[:, 0])
    right_total_score = sum(total_score[:, 1])
    relative_score = right_total_score - left_total_score

    print(f"{time.time() - st} seconds duration")
    print(f"left_total_score: {left_total_score}")
    print(f"right_total_score: {right_total_score}")
    print(f"relative_score: {relative_score}")

    if render:
        env.close()
    return relative_score,


def perform_episode(env, model1, model2, render):
    last_score = None
    action = np.copy(BLANK_ACTION)
    timeout_counter = 0
    while True:
        observation, reward, is_done, score_info = env.step(action)

        ball_location, left_location, right_location = find_stuff(observation)
        left_action = get_random_action(ALL_ACTIONS)
        right_action = get_random_action(ALL_ACTIONS)

        if ball_location is not None:
            if left_location is not None:
                left_action = inferrance(ball_location, left_location, right_location, model1)
            if right_location is not None:
                right_action = inferrance(ball_location, right_location, left_location, model2)

        left_action_restricted = keep_within_game_bounds_please(left_location, left_action)
        right_action_restricted = keep_within_game_bounds_please(right_location, right_action)

        action[RIGHT_ACTION_START:RIGHT_ACTION_END] = right_action_restricted
        action[RIGHT_ACTION_END:LEFT_ACTION_END] = left_action_restricted

        if last_score is not None:
            if last_score == score_info:
                timeout_counter += 1
            else:
                timeout_counter = 0
        last_score = score_info

        if render:
            env.render()
            time.sleep(1.0 / FPS)

        if is_done:
            break
        if timeout_counter > TIMEOUT_THRESH:
            break
    env.reset()
    # print(score_info)
    return [score_info["score1"], score_info["score2"]]


def get_random_action(all_actions):
    return all_actions[np.random.choice(all_actions.shape[0], size=None, replace=False), :]


def load_or_create_pop():
    list_of_files = glob.glob('checkpoints/*')
    if len(list_of_files) > 0:
        checkpoint = max(list_of_files, key=os.path.getctime)
    else:
        checkpoint = None

    if checkpoint:
        print("A file name has been given, then load the data from the file")
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        random.setstate(cp["rndstate"])
    else:
        print("Start a new evolution")
        population = toolbox.population(n=POPULATION_SIZE)
    return population


def main():
    population = load_or_create_pop()

    hall_of_fame = tools.HallOfFame(HALL_OF_FAME_AMOUNT)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    while True:
        population, log = algorithms.eaSimple(
            population, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=N_GENS,
            stats=stats, halloffame=hall_of_fame, verbose=True
        )

        print(log)
        print(hall_of_fame)

        save_checkpoint(population)


def save_checkpoint(population):
    cp = dict(
        population=population,
        rndstate=random.getstate()
    )
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/checkpoint_{datetime.datetime.now().strftime('%H_%M_%S')}.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)


if __name__ == '__main__':
    main()
