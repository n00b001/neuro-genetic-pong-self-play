import time

import retro
from deap import algorithms
from deap import tools

import ga
from dumb_ais import HardcodedAi, ScoreHardcodedAi
from ga import toolbox, hall_of_fame
from human_control import HumanInput
from utils import *
from utils import inference

""":arg action[0] is up, action[1] is down"""


def run_against_human(individual=None):
    right_model = create_model_from_genes(individual)
    left_model = HumanInput()

    env = retro.make(GAME_NAME, state='Start.2P', players=2)

    env.use_restricted_actions = retro.Actions.FILTERED
    env.reset()
    perform_episode(env, left_model, right_model, True, 1)


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
    action = np.copy(BLANK_ACTION)
    timeout_counter = 0.0
    total_frames = 0.0
    last_ball_location = None
    st = time.time()
    while True:
        observation, reward, is_done, score_info = env.step(action)
        if type(left_model) == ScoreHardcodedAi:
            left_model.set_score(score_info)

        ball_location, left_location, right_location = find_stuff_quickly(observation)
        del observation
        left_action, right_action = get_actions(
            ball_location, last_ball_location, left_location, left_model, right_location, right_model
        )
        last_ball_location = ball_location

        left_action_restricted = keep_within_game_bounds_please(left_location, left_action)
        right_action_restricted = keep_within_game_bounds_please(right_location, right_action)

        action[RIGHT_ACTION_START:RIGHT_ACTION_END] = right_action_restricted
        action[RIGHT_ACTION_END:LEFT_ACTION_END] = left_action_restricted

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
    right_reward = calculate_reward(score_multiplier, total_frames, score_info["score2"], score_info["score1"])
    return right_reward


def render_game(env, st):
    rgb = env.render('rgb_array')
    # gets choppy when scaled over 4,4 for me
    upscaled = repeat_upsample(rgb, 4, 4)
    viewer.imshow(upscaled)
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


def get_actions(ball_location, last_ball_location, left_location, left_model, right_location, right_model):
    left_action = get_random_action(ALL_ACTIONS)
    right_action = get_random_action(ALL_ACTIONS)
    if last_ball_location is None:
        last_ball_location = ball_location
    if ball_location is not None:
        if left_location is not None:
            # here we are flipping X
            left_ball_loc = [ball_location[0], GAME_WIDTH - ball_location[1]]
            left_last_ball_loc = [last_ball_location[0], GAME_WIDTH - last_ball_location[1]]
            left_action = inference(left_ball_loc, left_last_ball_loc, left_location, right_location, left_model)
        if right_location is not None:
            right_action = inference(ball_location, last_ball_location, right_location, left_location, right_model)
    else:
        left_action = [0, 0]
        right_action = [0, 0]
    return left_action, right_action


def main():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    while True:
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
