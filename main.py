import time

import retro
from deap import algorithms
from deap import tools
from gym.envs.classic_control import rendering

import ga
from dumb_ais import HardcodedAi, ScoreHardcodedAi
from ga import toolbox, hall_of_fame
from utils import *
from utils import inference

""":arg action[0] is up, action[1] is down"""


def evaluate(individual=None, render=RENDER):
    right_model = create_model_from_genes(individual)

    all_rewards = []
    right_score_multiplier = 1
    left_model = HardcodedAi()
    for i in range(GAMES_TO_PLAY):
        env = None
        try:
            if i == 0:
                pass
            elif i == 1:
                env = retro.make('Pong-Atari2600')
            elif i == 2:
                left_model = ScoreHardcodedAi()
            else:
                if hall_of_fame is None:
                    pass
                else:
                    left_model, right_score_multiplier = create_model_from_hall_of_fame(hall_of_fame)
            if env is None:
                env = retro.make('Pong-Atari2600', state='Start.2P', players=2)

            env.use_restricted_actions = retro.Actions.FILTERED
            env.reset()

            right_reward = perform_episode(env, left_model, right_model, render, right_score_multiplier)
            all_rewards.append(right_reward)
        finally:
            env.close()

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
                total_frames += timeout_counter
                timeout_counter = 0.0
        last_score = score_info

        if render:
            rgb = env.render('rgb_array')
            # gets choppy when scaled over 4,4 for me
            upscaled = repeat_upsample(rgb, 4, 4)
            viewer.imshow(upscaled)
            desired_sleep_time = 1.0 / FPS
            calculation_duration = time.time() - st
            actual_sleep_time = max(desired_sleep_time - calculation_duration, 0)
            time.sleep(actual_sleep_time)
            st = time.time()

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

        save_checkpoint(ga.population, hall_of_fame)


toolbox.register("evaluate", evaluate)
viewer = rendering.SimpleImageViewer()

if __name__ == '__main__':
    main()
    # individual = [3.0982564618623556, 0.3456392740138593, 0.14082890678241, 1.5486152183076243, 1.688957528292551, -0.41022947331564297, 1.9202332819872239, -0.8111005037794137, -2.1761033755921253, 0.0816563985669917, 1.237618565571571, 3.8602528621388834, 0.5584348334191216, 0.7207469622503221, -0.736499399144851, -2.0237233647139488, -0.08334005559341429, -0.8852364356431746, -1.2334751501020085, -1.623317909185873, 1.3730356225306743, -0.8823778108129245, 0.5819496473264711, 1.1509894218336085, -2.291795897101383, 0.7258951582377213, 2.573366867768538, -0.07424030006561644]
    # evaluate(individual=individual, render=True)
