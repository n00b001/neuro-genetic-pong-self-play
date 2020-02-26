import time

# import retro
from deap import algorithms
from deap import tools

import ga
from dumb_ais import HardcodedAi, ScoreHardcodedAi
from ga import toolbox, hall_of_fame
# from human_control import HumanInput
from numpy_nn import create_model_from_genes, create_model_from_hall_of_fame
from utils import *
from utils import get_actions

""":arg action[0] is up, action[1] is down"""


# @profile(precision=4)
def evaluate(individual=None, render=RENDER):
    right_model = create_model_from_genes(individual)

    all_rewards = []
    right_score_multiplier = 1
    for i in range(GAMES_TO_PLAY):
        env = GAME_2_PLAYER
        left_model = HardcodedAi()
        try:
            if i == 0:
                # HardcodedAi
                # GAME_2_PLAYER
                pass
            elif i == 1:
                # ScoreHardcodedAi
                # GAME_2_PLAYER
                left_model = ScoreHardcodedAi()
            else:
                if hall_of_fame is None:
                    pass
                else:
                    new_model, right_score_multiplier = create_model_from_hall_of_fame(hall_of_fame)
                    if new_model is not None:
                        left_model = new_model

            env.reset()

            right_reward = perform_episode(env, left_model, right_model, render, right_score_multiplier)
            all_rewards.append(right_reward)
        finally:
            # env.close()
            # del env
            del left_model

    average_reward = sum(all_rewards) / float(GAMES_TO_PLAY)
    return average_reward,


def perform_episode(env, left_model, right_model, render, score_multiplier):
    last_score = None
    actions = {
        "player1":Direction.NOOP,
        "player2":Direction.NOOP
    }
    timeout_counter = 0.0
    total_frames = 0.0
    last_ball_location = None
    st = time.time()
    while True:
        observation, _, is_done, score_info = env.step(actions)
        if type(left_model) == ScoreHardcodedAi:
            left_model.set_score(score_info)

        # ball_location, left_location, right_location = find_stuff_quickly(observation)
        # del observation
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
            st = render_game(env, st)

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


def render_game(env, st):
    env.render()
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
# GAME_2_PLAYER.start()

toolbox.register("evaluate", evaluate)
try:
    from gym.envs.classic_control import rendering

    viewer = rendering.SimpleImageViewer()
except Exception as e:
    scoop.logger.error(e)

if __name__ == '__main__':
    main()
