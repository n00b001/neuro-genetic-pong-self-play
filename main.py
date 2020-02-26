import time

from deap import algorithms
from deap import tools

import ga
from dumb_ais import ScoreHardcodedAi, RandomHardcodedAi
from ga import toolbox, hall_of_fame
from my_intense_pong import MyPong
from numpy_nn import create_model_from_genes, create_model_from_hall_of_fame
from utils import *
from utils import get_actions

""":arg action[0] is up, action[1] is down"""


# @profile(precision=4)
def evaluate(individual=None, render=RENDER):
    right_model = create_model_from_genes(individual)

    all_rewards = []
    for i in range(GAMES_TO_PLAY):
        try:
            random_thresh = i * (1 / float(NUMBER_OF_HARD_CODED_RANDOM_AIS))
            left_model = RandomHardcodedAi(random_thresh=random_thresh)
            right_score_multiplier = random_thresh

            if random_thresh > 1 and hall_of_fame is not None:
                new_model, right_score_multiplier = create_model_from_hall_of_fame(hall_of_fame)
                if new_model is not None:
                    left_model = new_model

            ENV.reset()

            right_reward = perform_episode(ENV, left_model, right_model, render, right_score_multiplier)
            all_rewards.append(right_reward)
        finally:
            del left_model

    average_reward = sum(all_rewards) / float(GAMES_TO_PLAY)
    return average_reward,


def perform_episode(env, left_model, right_model, render, score_multiplier):
    last_score = None
    actions = {
        "player1": Direction.NOOP,
        "player2": Direction.NOOP
    }
    timeout_counter = 0.0
    total_frames = 0.0
    last_ball_location = None
    st = time.time()
    while True:
        observation, _, is_done, score_info = env.step(actions)
        if type(left_model) == ScoreHardcodedAi:
            left_model.set_score(score_info)
        if type(right_model) == ScoreHardcodedAi:
            right_model.set_score(score_info)

        left_action, right_action = get_actions(
            observation[0], last_ball_location, observation[1], left_model, observation[2], right_model
        )
        last_ball_location = observation[0]

        actions["player1"] = left_action
        actions["player2"] = right_action

        timeout_counter, total_frames = calculate_timeout_and_frames(
            last_score, score_info, timeout_counter, total_frames
        )
        last_score = score_info

        if render:
            st = render_game(env, st, left_model, right_model)

        if score_info["score1"] >= WIN_SCORE or score_info["score2"] >= WIN_SCORE:
            break
        if is_done:
            break
        if timeout_counter > TIMEOUT_THRESH:
            break
    env.reset()
    if score_info["score1"] == score_info["score2"]:
        return 0
    total_frames += timeout_counter
    right_reward = calculate_reward(score_multiplier, total_frames, score_info["score2"], score_info["score1"])
    return right_reward


def render_game(env, st, left_class=None, right_class=None):
    env.render(left_class, right_class)
    # gets choppy when scaled over 4,4 for me
    # upscaled = repeat_upsample(rgb, 4, 4)
    # viewer.imshow(upscaled)
    desired_sleep_time = 1.0 / FPS
    calculation_duration = time.time() - st
    actual_sleep_time = max(desired_sleep_time - calculation_duration, 0)
    time.sleep(actual_sleep_time)
    st = time.time()
    return st


def calculate_timeout_and_frames(last_score, score_info, timeout_counter, total_frames):
    if last_score is not None:
        if last_score == score_info:
            timeout_counter += 1.0
        else:
            total_frames += timeout_counter
            timeout_counter = 0.0
    return timeout_counter, total_frames


def main():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    ga.population, log = algorithms.eaSimple(
        ga.population, toolbox,
        cxpb=CROSSOVER_BLEND_PROBABILITY, mutpb=GAUSSIAN_MUTATION_PROBABILITY,
        ngen=GENERATIONS_BEFORE_SAVE,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )
    # scoop.logger.info(log)

    save_checkpoint(ga.population, hall_of_fame)
    del log


# GAME_1_PLAYER.start()
# GAME_2_PLAYER.start(

ENV = MyPong(RENDER)
toolbox.register("evaluate", evaluate)
try:
    from gym.envs.classic_control import rendering

    viewer = rendering.SimpleImageViewer()
except Exception as e:
    scoop.logger.error(e)

if __name__ == '__main__':
    main()
