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

FPS = 60
SCALE_FACTOR = 2
MAX_STEPS = 100


def find_stuff(observation):
    # ball_marks = cv2.inRange(observation, BALL_COLOUR - RANGE, )
    small_img = cv2.resize(
        observation,
        (observation.shape[1] // SCALE_FACTOR, observation.shape[0] // SCALE_FACTOR),
        interpolation=cv2.INTER_NEAREST
    )
    chopped_observation = small_img[34 // SCALE_FACTOR:194 // SCALE_FACTOR, :]
    ball_loc = get_rect(chopped_observation, BALL_COLOUR)
    lefty_loc = get_rect(chopped_observation, LEFT_GUY_COLOUR)
    far_right_scum = get_rect(chopped_observation, RIGHT_GUY_COLOUR)
    return ball_loc, lefty_loc, far_right_scum


def get_rect(chopped_observation, colour):
    indices = np.where(np.all(chopped_observation == colour, axis=-1))
    return np.average(indices, axis=1)


def inferrance(ball_loc, lefty_loc, far_right_scum):
    return []


def main(render=True):
    env = retro.make('Pong-Atari2600', state='Start.2P', players=2)
    observation = env.reset()
    for i in range(MAX_STEPS):
        observation, reward, is_done, score_info = env.step(env.action_space.sample())

        ball_loc, lefty_loc, far_right_scum = find_stuff(observation)
        # action_1 = inferrance(ball_loc, lefty_loc, far_right_scum, model1)
        # action_2 = inferrance(ball_loc, lefty_loc, far_right_scum, model2)
        # action = action_1 + action_2

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
