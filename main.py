import datetime
import glob
import os
import pickle
import random
import time
from copy import deepcopy

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

toolbox.register("mate", tools.cxBlend, alpha=CX_ALPHA)
# toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=IND_PB)
# toolbox.register("select", tools.selLexicase, k=TOURN_SIZE)
# toolbox.register("select", tools.selBest, TOURN_SIZE)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

toolbox.register("evaluate", eval)

hall_of_fame = None


def find_stuff(observation):
    chopped_observation = observation[GAME_TOP: GAME_BOTTOM, :]
    ball_location = get_rect(chopped_observation, BALL_COLOUR)
    left_location = get_rect(chopped_observation, LEFT_GUY_COLOUR)
    right_location = get_rect(chopped_observation, RIGHT_GUY_COLOUR)
    return ball_location, left_location, right_location


def get_rect(chopped_observation, colour):
    indices = np.where(np.all(chopped_observation == colour, axis=-1))
    if len(indices[0]) == 0:
        return None
    return np.average(indices, axis=1)


def inference(ball_location, last_ball_location, me, enemy, model):
    normalised_ball_location_x = ball_location[1] / GAME_WIDTH
    normalised_ball_location_y = ball_location[0] / GAME_PLAYABLE_HEIGHT
    normalised_last_ball_location_x = last_ball_location[1] / GAME_WIDTH
    normalised_last_ball_location_y = last_ball_location[0] / GAME_PLAYABLE_HEIGHT
    normalised_me = me[0] / GAME_PLAYABLE_HEIGHT
    normalised_enemy = enemy[0] / GAME_PLAYABLE_HEIGHT
    predictions = model.run(
        [
            normalised_ball_location_x, normalised_ball_location_y,
            normalised_last_ball_location_x,
            normalised_last_ball_location_y,
            normalised_me, normalised_enemy
        ])
    return predictions


""":arg action[0] is up, action[1] is down"""


def keep_within_game_bounds_please(paddle, action):
    if paddle is not None:
        if paddle[0] < SCALED_PADDLE_HEIGHT:
            action = [0, 1]
        elif paddle[0] > ((GAME_BOTTOM - GAME_TOP) - SCALED_PADDLE_HEIGHT):
            action = [1, 0]
    return action


def load_model_from_genes(individual):
    simple_network = NeuralNetwork(
        nodes=NETWORK_SHAPE,
        weights=individual,
        bias=True
    )

    return simple_network


class HardcodedAi:
    def run(self, input_vector):
        ret_val = [0, 0]
        if input_vector[1] < input_vector[4]:
            ret_val[0] = 1
        elif input_vector[1] > input_vector[4]:
            ret_val[1] = 1
        return ret_val


class ScoreHardcodedAi:
    def __int__(self):
        self.score_info = {}

    def set_score(self, score_info):
        self.score_info = score_info

    def run(self, input_vector):
        ret_val = [0, 0]
        if self.score_info["score1"] <= self.score_info["score2"]:
            if input_vector[1] < input_vector[4]:
                ret_val[0] = 1
            elif input_vector[1] > input_vector[4]:
                ret_val[1] = 1
        return ret_val


def evaluate(individual=None, render=False):
    right_model = load_model_from_genes(individual)

    st = time.time()
    total_score = []
    for i in range(GAMES_TO_PLAY):
        env = None
        if i == 0:
            left_model = HardcodedAi()
        elif i == 1:
            left_model = HardcodedAi()
            env = retro.make('Pong-Atari2600')
        elif i == 2:
            left_model = ScoreHardcodedAi()
        else:
            if hall_of_fame is None or len(hall_of_fame.items) == 0:
                left_model = HardcodedAi()
            else:
                left_model = load_model_from_genes(random.choice(hall_of_fame.items))
        if env is None:
            env = retro.make('Pong-Atari2600', state='Start.2P', players=2)

        env.use_restricted_actions = retro.Actions.FILTERED
        env.reset()

        score_info = perform_episode(env, left_model, right_model, render)
        total_score.append(score_info)

        env.close()

    total_score = np.array(total_score)
    left_total_score = sum(total_score[:, 0])
    right_total_score = sum(total_score[:, 1])
    average_score = right_total_score / float(GAMES_TO_PLAY)

    # print(f"{time.time() - st} seconds duration")
    # print(f"left_total_score: {left_total_score}")
    # print(f"right_total_score: {right_total_score}")
    # print(f"relative_score: {relative_score}")

    # env.close()
    return average_score,


def perform_episode(env, left_model, right_model, render):
    last_score = None
    action = np.copy(BLANK_ACTION)
    timeout_counter = 0.0
    total_time = 0.0
    last_ball_location = None
    while True:
        observation, reward, is_done, score_info = env.step(action)
        if type(left_model) == ScoreHardcodedAi:
            left_model.set_score(score_info)

        ball_location, left_location, right_location = find_stuff(observation)
        left_action = get_random_action(ALL_ACTIONS)
        right_action = get_random_action(ALL_ACTIONS)

        if last_ball_location is None:
            last_ball_location = ball_location

        if ball_location is not None:
            if left_location is not None:
                left_ball_loc = ball_location
                left_ball_loc[1] = GAME_WIDTH - left_ball_loc[1]
                left_last_ball_loc = last_ball_location
                left_last_ball_loc[1] = GAME_WIDTH - left_last_ball_loc[1]
                left_action = inference(left_ball_loc, left_last_ball_loc, left_location, right_location, left_model)
            if right_location is not None:
                right_action = inference(ball_location, last_ball_location, right_location, left_location, right_model)
        else:
            left_action = [0, 0]
            right_action = [0, 0]

        last_ball_location = ball_location

        left_action_restricted = keep_within_game_bounds_please(left_location, left_action)
        right_action_restricted = keep_within_game_bounds_please(right_location, right_action)

        action[RIGHT_ACTION_START:RIGHT_ACTION_END] = right_action_restricted
        action[RIGHT_ACTION_END:LEFT_ACTION_END] = left_action_restricted

        if last_score is not None:
            if last_score == score_info:
                timeout_counter += 1.0
            else:
                total_time += timeout_counter
                timeout_counter = 0.0
        last_score = score_info

        if render:
            env.render()
            time.sleep(1.0 / FPS)

        if score_info["score1"] >= WIN_SCORE or score_info["score2"] >= WIN_SCORE:
            break
        if is_done:
            break
        if timeout_counter > TIMEOUT_THRESH:
            break
    env.reset()
    # print(score_info)
    if score_info["score1"] == score_info["score2"]:
        return [0, 0]

    left_reward = (score_info["score1"] - score_info["score2"]) / (total_time / 1000.0)
    right_reward = (score_info["score2"] - score_info["score1"]) / (total_time / 1000.0)
    return [left_reward, right_reward]


def get_random_action(all_actions):
    return all_actions[np.random.choice(all_actions.shape[0], size=None, replace=False), :]


def load_population():
    global hall_of_fame
    list_of_files = glob.glob('checkpoints/*')
    if len(list_of_files) > 0:
        checkpoint = max(list_of_files, key=os.path.getctime)
        print("A file name has been given, then load the data from the file")
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        _population = cp["population"]
        _sorted_population = sorted(_population, key=lambda x: x.fitness.values[0], reverse=True)
        random.setstate(cp["rndstate"])
        if "hall_of_fame" in cp:
            hall_of_fame = cp["hall_of_fame"]
        return _sorted_population
    else:
        return None


def load_or_create_pop():
    global hall_of_fame

    population = load_population()

    loaded_pop_size = 0
    if population is not None:
        loaded_pop_size = len(population)

    if loaded_pop_size >= POPULATION_SIZE:
        return population[:POPULATION_SIZE]

    _population = toolbox.population(n=(POPULATION_SIZE - loaded_pop_size))

    if loaded_pop_size == 0:
        return _population
    return population + _population


def main():
    global hall_of_fame

    hall_of_fame = tools.HallOfFame(HALL_OF_FAME_AMOUNT)
    population = load_or_create_pop()

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
        hall_of_fame=deepcopy(hall_of_fame),
        rndstate=random.getstate()
    )
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/checkpoint_{datetime.datetime.now().strftime('%H_%M_%S')}.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)


if __name__ == '__main__':
    main()
    # individual = [3.0982564618623556, 0.3456392740138593, 0.14082890678241, 1.5486152183076243, 1.688957528292551, -0.41022947331564297, 1.9202332819872239, -0.8111005037794137, -2.1761033755921253, 0.0816563985669917, 1.237618565571571, 3.8602528621388834, 0.5584348334191216, 0.7207469622503221, -0.736499399144851, -2.0237233647139488, -0.08334005559341429, -0.8852364356431746, -1.2334751501020085, -1.623317909185873, 1.3730356225306743, -0.8823778108129245, 0.5819496473264711, 1.1509894218336085, -2.291795897101383, 0.7258951582377213, 2.573366867768538, -0.07424030006561644]
    # evaluate(individual=individual, render=True)
