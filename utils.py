import datetime
import os
import pickle
import random
from copy import deepcopy

import scoop

from config import *
from config import GAME_WIDTH, GAME_PLAYABLE_HEIGHT
from numpy_nn import NeuralNetwork


def find_stuff(observation):
    chopped_observation = observation[GAME_TOP: GAME_BOTTOM, :]
    ball_location = get_rect_quickly(chopped_observation, BALL_COLOUR)
    left_location = get_rect_quickly(chopped_observation, LEFT_GUY_COLOUR)
    right_location = get_rect_quickly(chopped_observation, RIGHT_GUY_COLOUR)
    return np.array([ball_location, left_location, right_location])


# @np.vectorize
def find_stuff_quickly(observation):
    chopped_observation = observation[GAME_TOP: GAME_BOTTOM, :]
    locations = np.argwhere([
        chopped_observation == BALL_COLOUR,
        chopped_observation == LEFT_GUY_COLOUR,
        chopped_observation == RIGHT_GUY_COLOUR
    ])[:, :-1]

    split = np.split(
        locations[:, 1:], np.cumsum(np.unique(locations[:, 0], return_counts=True)[1])[:-1]
    )

    if len(split) < 2:
        return np.array([None, None, None])

    ball_location = np.average(split[0], axis=0)
    left_location = np.average(split[1], axis=0)
    right_location = np.average(split[2], axis=0)

    del split
    del locations
    del chopped_observation

    # https://stackoverflow.com/questions/911871/detect-if-a-numpy-array-contains-at-least-one-non-numeric-value
    if np.isnan(ball_location).any() or np.isnan(left_location).any() or np.isnan(right_location).any():
        return np.array([None, None, None])
    return np.array([ball_location, left_location, right_location])


def get_rect(chopped_observation, colour):
    indices = np.where(np.all(chopped_observation == colour, axis=-1))
    if len(indices[0]) == 0:
        return None
    value = np.average(indices, axis=1)
    return value


def get_rect_quickly(chopped_observation, colour):
    value = np.average(
        np.argwhere(chopped_observation == colour)[:, :-1],
        axis=0
    )
    # https://stackoverflow.com/questions/911871/detect-if-a-numpy-array-contains-at-least-one-non-numeric-value
    if np.isnan(value).any():
        return None
    return value


def keep_within_game_bounds_please(paddle, action):
    if paddle is not None:
        if paddle[0] < SCALED_PADDLE_HEIGHT:
            action = [0, 1]
        elif paddle[0] > ((GAME_BOTTOM - GAME_TOP) - SCALED_PADDLE_HEIGHT):
            action = [1, 0]
    return action


def create_model_from_genes(individual):
    simple_network = NeuralNetwork(
        nodes=NETWORK_SHAPE,
        weights=individual,
        bias=BIAS
    )

    return simple_network


def create_model_from_hall_of_fame(hall_of_fame):
    left_model = None
    right_score_multiplier = 1
    hall_of_fame_items = hall_of_fame.items
    if len(hall_of_fame_items) != 0:
        random.shuffle(hall_of_fame_items)
        for hall_of_famer in hall_of_fame_items:
            if hall_of_famer.fitness.valid:
                right_score_multiplier = hall_of_famer.fitness.values[0]
                left_model = create_model_from_genes(list(hall_of_famer))
                break
    return left_model, right_score_multiplier


def calculate_reward(score_multiplier, total_time, my_score, enemy_score):
    diff = my_score - enemy_score
    scaled_time = total_time / TIME_SCALER
    bonus_points = my_score * score_multiplier
    reward = (diff + bonus_points) / scaled_time
    return reward


def get_random_action(all_actions):
    return all_actions[np.random.choice(all_actions.shape[0], size=None, replace=False), :]


def save_checkpoint(_population, hall_of_fame):
    cp = dict(
        population=_population,
        hall_of_fame=deepcopy(hall_of_fame),
        rndstate=random.getstate(),
        network_shape=NETWORK_SHAPE
    )
    os.makedirs("checkpoints/checkpoints", exist_ok=True)
    with open("checkpoints/checkpoints/c_{}.pkl".format(datetime.datetime.now().strftime('%H_%M_%S')), "wb") as cp_file:
        pickle.dump(cp, cp_file)


def calculate_gene_size():
    total_genes = 0
    for i in range(len(NETWORK_SHAPE) - 1):
        input_node_amount = NETWORK_SHAPE[i]
        output_node_amount = NETWORK_SHAPE[i + 1]
        bias = 1 if BIAS else 0
        genes = (input_node_amount + bias) * output_node_amount
        total_genes += genes
    return total_genes


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


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            scoop.logger.error(
                "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
