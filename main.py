import random
import time

import cv2
import numpy as np
import retro

# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3

BG_COLOUR = (144, 72, 17)
BALL_COLOUR = (236, 236, 236)
LEFT_GUY_COLOUR = (213, 130, 74)
RIGHT_GUY_COLOUR = (92, 186, 92)

FPS = 600
SCALE_FACTOR = 2
MAX_STEPS = 100_000
GAME_BOTTOM = 194
GAME_TOP = 34
SCALED_PADDLE_HEIGHT = 16.0 / SCALE_FACTOR

RIGHT_ACTION_START = 4
RIGHT_ACTION_END = 6
LEFT_ACTION_END = 8
LEFT_PLAYER_START_BUTTON = -1
RIGHT_PLAYER_START_BUTTON = 0


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
    if me is not None:
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


# action = [
#     1,  # maybe start game?
#     0,  # nothing
#     0,
#     0,
#     0,  # right, up
#     0,  # right, down
#     0,  # left, up
#     0  # left, down
# ]
def main(render=True):
    env = retro.make('Pong-Atari2600', state='Start.2P', players=2)
    # env = retro.make('Pong-Atari2600')#, state='Start.2P', players=2)
    env.use_restricted_actions = retro.Actions.ALL
    sample_actions = env.action_space.sample()

    observation = env.reset()
    action = np.zeros(shape=(16,), dtype=np.int)
    action[LEFT_PLAYER_START_BUTTON] = 1
    action[RIGHT_PLAYER_START_BUTTON] = 1

    action[LEFT_ACTION_END:-1] = 1
    model1, model2 = None, None
    for i in range(MAX_STEPS):
        observation, reward, is_done, score_info = env.step(action)

        ball_location, left_location, right_location = find_stuff(observation)
        if ball_location is None or np.isnan(ball_location[0]):
            left_action = [random.randint(0, 1), random.randint(0, 1)]
            right_action = [random.randint(0, 1), random.randint(0, 1)]
        else:
            left_action = inferrance(ball_location, left_location, right_location, model1)
            right_action = inferrance(ball_location, right_location, left_location, model2)

        left_action_restricted = keep_within_game_bounds_please(left_location, left_action)
        right_action_restricted = keep_within_game_bounds_please(right_location, right_action)

        action[RIGHT_ACTION_START:RIGHT_ACTION_END] = right_action_restricted
        action[RIGHT_ACTION_END:LEFT_ACTION_END] = left_action_restricted

        if render:
            env.render()

        # if the episode is over, reset the environment
        if is_done:
            observation = env.reset()
        time.sleep(1.0 / FPS)
    if render:
        env.close()


if __name__ == '__main__':
    main()
