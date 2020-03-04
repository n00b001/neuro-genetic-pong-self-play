import datetime
import math
import os
import pickle
import random

import numpy as np

from consts import Direction, NETWORK_SHAPE, BIAS, BONUS_SCALAR, GAME_HEIGHT, GAME_WIDTH, WIN_SCORE


def calculate_reward(score_multiplier, my_score, enemy_score):
    # diff = my_score - enemy_score

    # this means we get a big positive bonus if we win (10, -10)
    # this means we get a bonus of 0 if we just get bodied (-10, 10)
    bonus_points = ((my_score - enemy_score) + (WIN_SCORE * 2.0)) * score_multiplier
    reward = my_score + (bonus_points / BONUS_SCALAR)
    return reward


def get_random_action(all_actions):
    return all_actions[np.random.choice(all_actions.shape[0], size=None, replace=False), :]


def get_random_action2(all_actions):
    return all_actions[random.randint(0, len(all_actions) - 1)]


def get_random_action3():
    return random.choice(list(Direction))


def save_checkpoint(_population, hall_of_fame):
    cp = dict(
        population=_population,
        hall_of_fame=hall_of_fame,
        rndstate=random.getstate(),
        network_shape=NETWORK_SHAPE
    )
    os.makedirs("checkpoints/checkpoints", exist_ok=True)
    with open("checkpoints/checkpoints/c_{}.pkl".format(datetime.datetime.now().strftime('%H_%M_%S')), "wb") as cp_file:
        pickle.dump(cp, cp_file)
    del cp


def calculate_gene_size():
    total_genes = 0
    for i in range(len(NETWORK_SHAPE) - 1):
        input_node_amount = NETWORK_SHAPE[i]
        output_node_amount = NETWORK_SHAPE[i + 1]
        bias = 1 if BIAS else 0
        genes = (input_node_amount + bias) * output_node_amount
        total_genes += genes
    return total_genes


def inference(ball_location, last_ball_location, last_last_ball_location, me, enemy, model):
    normalised_ball_location_x = ball_location[1] / GAME_WIDTH
    normalised_ball_location_y = ball_location[0] / GAME_HEIGHT
    normalised_last_ball_location_x = last_ball_location[1] / GAME_WIDTH
    normalised_last_ball_location_y = last_ball_location[0] / GAME_HEIGHT
    normalised_last_last_ball_location_x = last_last_ball_location[1] / GAME_WIDTH
    normalised_last_last_ball_location_y = last_last_ball_location[0] / GAME_HEIGHT
    normalised_me = me[0] / GAME_HEIGHT
    normalised_enemy = enemy[0] / GAME_HEIGHT
    predictions = model.run(
        [
            normalised_ball_location_x, normalised_ball_location_y,
            normalised_last_ball_location_x, normalised_last_ball_location_y,
            normalised_last_last_ball_location_x, normalised_last_last_ball_location_y,
            normalised_me, normalised_enemy
        ])
    return predictions


def get_actions(observations, left_model, right_model):
    ball_location = observations[0]
    last_ball_location = observations[1]
    last_last_ball_location = observations[2]
    left_location = observations[3]
    right_location = observations[4]

    left_action = get_random_action3()
    right_action = get_random_action3()
    if ball_location is not None:
        if left_location is not None:
            # here we are flipping X
            left_ball_loc = [ball_location[0], GAME_WIDTH - ball_location[1]]
            left_last_ball_loc = [last_ball_location[0], GAME_WIDTH - last_ball_location[1]]
            left_last_last_ball_loc = [last_last_ball_location[0], GAME_WIDTH - last_last_ball_location[1]]
            left_action = inference(
                left_ball_loc, left_last_ball_loc, left_last_last_ball_loc, left_location, right_location, left_model
            )
        if right_location is not None:
            right_action = inference(
                ball_location, last_ball_location, last_last_ball_location, right_location, left_location, right_model
            )
    else:
        left_action = Direction.NOOP
        right_action = Direction.NOOP
    return left_action, right_action


def rotate(angle, point, origin=(0, 0)):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]


# @np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
