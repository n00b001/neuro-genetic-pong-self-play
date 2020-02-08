import datetime
import glob
import os
import pickle
import random
import time

import cv2
import retro
from deap import base, algorithms
from deap import creator
from deap import tools
from scoop import futures

from config import *
from numpy_nn import NeuralNetwork


def eval(individual):
    return evaluate(individual, RENDER)


creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)

toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENE_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=IND_PB)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

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


def inferrance(ball_location, me, enemy, model: NeuralNetwork):
    if model is not None:
        normalised_ball_location_x = ball_location[1] / GAME_WIDTH
        normalised_ball_location_y = ball_location[0] / GAME_PLAYABLE_HEIGHT
        normalised_me = me[0] / GAME_PLAYABLE_HEIGHT
        normalised_enemy = enemy[0] / GAME_PLAYABLE_HEIGHT
        predictions = model.run(
            [normalised_ball_location_x, normalised_ball_location_y, normalised_me, normalised_enemy])
        inx = np.argmax(predictions)
        return_arr = np.zeros(shape=predictions.shape, dtype=np.int)
        return_arr[inx] = 1
        return return_arr
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
        elif paddle[0] > (((GAME_BOTTOM - GAME_TOP) / SCALE_FACTOR) - SCALED_PADDLE_HEIGHT):
            action = [1, 0]
    return action


def load_model_from_genes(individual):
    nodes = [4, 4, 2]

    simple_network = NeuralNetwork(
        nodes=nodes,
        weights=individual
    )

    return simple_network


def evaluate(individual=None, render=False):
    # env = retro.make('Pong-Atari2600', state='Start.2P', players=2)
    env = retro.make('Pong-Atari2600')
    env.use_restricted_actions = retro.Actions.FILTERED
    env.reset()

    left_model = None
    right_model = load_model_from_genes(individual)

    st = time.time()
    total_score = []
    for i in range(GAMES_TO_PLAY):
        score_info = perform_episode(env, left_model, right_model, render)
        total_score.append(score_info)

    total_score = np.array(total_score)
    left_total_score = sum(total_score[:, 0])
    right_total_score = sum(total_score[:, 1])
    relative_score = right_total_score - left_total_score
    average_score = relative_score / float(GAMES_TO_PLAY)

    # print(f"{time.time() - st} seconds duration")
    # print(f"left_total_score: {left_total_score}")
    # print(f"right_total_score: {right_total_score}")
    # print(f"relative_score: {relative_score}")

    if render:
        env.close()
    return average_score,


def perform_episode(env, left_model, right_model, render):
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
                left_action = inferrance(ball_location, left_location, right_location, left_model)
            if right_location is not None:
                right_action = inferrance(ball_location, right_location, left_location, right_model)

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
    # individual = [
    #     0.45166043246976356, 1.0131873834425835, 0.4548966304383305, 1.049285360057891, 0.04790764045279524, -0.04799704016690026, 0.3759395605860322, -0.0328326609370036, 0.36523007241571503, 0.030259432842876938, 0.7775061965238808, 0.25896916543285564, 0.6931445173729602, 0.8506462374766993, 0.04424946622071191, 0.9408616565546967, 0.13132296262550366, 0.7940050269702555, 0.8645860937500492, 0.6886954843500778, 0.7122791817651364, 0.3164679781875281, 0.3999678493264221, 0.8779594039734829
    # ]
    # evaluate(individual=individual, render=True)
