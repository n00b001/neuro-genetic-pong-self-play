import datetime
import glob
import os
import pickle
import random
import time

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


def inference(ball_location, last_ball_location, me, enemy, model: NeuralNetwork):
    if model is not None:
        normalised_ball_location_x = ball_location[1] / GAME_WIDTH
        normalised_ball_location_y = ball_location[0] / GAME_PLAYABLE_HEIGHT
        normalised_last_ball_location_x = last_ball_location[1] / GAME_WIDTH
        normalised_last_ball_location_y = last_ball_location[0] / GAME_PLAYABLE_HEIGHT
        normalised_me = me[0] / GAME_PLAYABLE_HEIGHT
        normalised_enemy = enemy[0] / GAME_PLAYABLE_HEIGHT
        predictions = model.run(
            [
                normalised_ball_location_x, normalised_ball_location_y, normalised_last_ball_location_x,
                normalised_last_ball_location_y, normalised_me, normalised_enemy
            ])
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
        elif paddle[0] > ((GAME_BOTTOM - GAME_TOP) - SCALED_PADDLE_HEIGHT):
            action = [1, 0]
    return action


def load_model_from_genes(individual):
    simple_network = NeuralNetwork(
        nodes=NETWORK_SHAPE,
        weights=individual,
    )

    return simple_network


def evaluate(individual=None, render=False):
    right_model = load_model_from_genes(individual)

    st = time.time()
    total_score = []
    build_in_ai = False
    left_model = None
    for i in range(GAMES_TO_PLAY):
        random_num = random.random()

        # if 0 < random_num < 0.4:  # 40% of time we'll use hardcoded AI as enemy
        if i == 0:
            left_model = None
        # elif 0.4 < random_num < 0.8:  # 40% of the time we'll use built in AI as enemy
        elif i == 1:
            build_in_ai = True
        else:  # 20% of the time, we'll use one of the hof individuals
            build_in_ai = False
            if hall_of_fame is None or len(hall_of_fame.items) == 0:
                left_model = None
            else:
                left_model = load_model_from_genes(random.choice(hall_of_fame.items))

        if not build_in_ai:
            env = retro.make('Pong-Atari2600', state='Start.2P', players=2)
        else:
            env = retro.make('Pong-Atari2600')
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

        ball_location, left_location, right_location = find_stuff(observation)
        left_action = get_random_action(ALL_ACTIONS)
        right_action = get_random_action(ALL_ACTIONS)

        if last_ball_location is None:
            last_ball_location = ball_location

        if ball_location is not None:
            if left_location is not None:
                left_action = inference(ball_location, last_ball_location, left_location, right_location, left_model)
            if right_location is not None:
                right_action = inference(ball_location, last_ball_location, right_location, left_location, right_model)
        else:
            left_action = [0, 0]
            right_action = [0, 0]

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


def load_or_create_pop():
    global hall_of_fame
    list_of_files = glob.glob('checkpoints/*')
    if len(list_of_files) > 0:
        checkpoint = max(list_of_files, key=os.path.getctime)
    else:
        checkpoint = None

    population = toolbox.population(n=POPULATION_SIZE)
    if checkpoint:
        print("A file name has been given, then load the data from the file")
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        _population = cp["population"]
        _sorted_population = sorted(_population, key=lambda x: x.fitness.values[0], reverse=True)
        if len(_population) < POPULATION_SIZE:
            population[:len(_sorted_population)] = _sorted_population
        else:
            population = _sorted_population[:POPULATION_SIZE]
        random.setstate(cp["rndstate"])
        if "hall_of_fame" in cp:
            hall_of_fame = cp["hall_of_fame"]
    else:
        print("Start a new evolution")
    return population


def main():
    global hall_of_fame

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
        hall_of_fame=hall_of_fame,
        rndstate=random.getstate()
    )
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/checkpoint_{datetime.datetime.now().strftime('%H_%M_%S')}.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)


if __name__ == '__main__':
    # main()
    # individual = [2.545363772443644, -0.021478489313068172, -0.15366547839839745, 2.2505708563781006, -1.2229583072149612, 2.370732127380614, 1.2834865567640077, -2.981773714254297, -0.3856860847071438, -2.8513748632243803, -0.5535750224838425, -0.24154988683950307, 0.07362185729879138, -3.2389678818866274, -0.28246083169684616, -2.086782570362913, 1.7096264968292458, 1.493310535766563, 0.1698582390357819, -0.6405005313525163, -0.4922482492391475, 4.1526750716036585, -0.2803448936488474, 1.5037531566763553, -0.20685571536776814, -1.7921798717841417, 4.44907299301761, 0.4570858980011814, 1.9739009883039436, -0.916882354199865, -0.6480184564861109, 0.1309919863704414]
    # individual = [1.068020735845282, -0.3837447566977138, -1.0151046852036982, 0.6173748399639936, -0.24544238807571173, 1.7516867117389516, 3.323259961305144, -0.6833667271955816, -1.0844893680411845, 1.2182520395868042, -0.533879507175848, 3.5806929515214545, 0.32603860425215625, -0.14951238268445488, 0.6838495947305132, 0.46071333975996875, 0.8452268762629406, -0.3938545667368471, -0.32513719831913457, 4.601566605380813, 2.02284407743116, -2.846197358107322, 0.9722175954189243, 2.083247239594264, 2.516504059222358, 1.5361163299530791, 1.1326886432554684, 3.1430804027681605, -2.231738100436152, -0.6033534013225744, -1.5450442710869425, 0.8770036601130415, -0.15925316580178706, -0.31731699126486645, 2.610097450772107, -0.9320634828498745, 1.306111794921446, -1.1614864771830602, 0.6985574046963744, 2.2447804191077, 0.8335905141691349, -0.519506612813645, -0.5652313419972996, 2.246817129399486, -2.0005944713440664, -2.13570941596382, -0.3756279603673911, -0.8107836283944765, -1.2226955399360149, -1.2285193711916933, -1.391042148952684, -0.013968278620536179, -0.6354772524643483, 1.8806809661509496, 1.8044597773284012, 2.869460977590903]
    individual = [-0.7045202405715212, -0.6008990769410288, -0.6465487803498704, 3.72254501207871, 11.662591769226681,
                  7.286295970365752, 0.3055821312996996, -4.3094516893099515, -2.4567551342813614, -2.902325439446411,
                  -1.5324860245208896, 2.700125800335014, -2.231095034366262, -5.01830524838203, 3.2037271893351718,
                  1.1974324837177017, 5.148555207855356, 7.665746326690054, 4.550916833158775, -0.21529754902717024,
                  1.855545447145384, 10.220527070481713, -3.9641611516878488, -2.4698952671888033, 0.7135973371420858,
                  -0.1447860773844818, 3.657726460417585, 2.103004754770878, -2.0478831192986244, -5.1412366650336105,
                  4.654713600369238, -0.5952741867650442, 4.609814431154118, 3.328180644339965, -0.2753887395989051,
                  1.011678140199655, -0.3986237062078002, 2.147880868013523, 0.5000535470903792, -0.26750512772333085,
                  -4.956726076811883, -1.052104549416598, -6.912910915623136, 2.005566742267521, -0.04425203005992842,
                  -2.31810066108435]
    evaluate(individual=individual, render=True)
