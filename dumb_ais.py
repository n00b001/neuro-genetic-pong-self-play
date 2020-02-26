import random

from utils import Direction


class RandomHardcodedAi:
    def __init__(self, random_thresh):
        self.random_thresh = max(0, min(random_thresh, 1.0))

    def run(self, input_vector):
        """
        :param input_vector: [
            normalised_ball_location_x,
            normalised_ball_location_y,
            normalised_last_ball_location_x,
            normalised_last_ball_location_y,
            normalised_me,
            normalised_enemy
        ]
        :type input_vector:
        :return: action[0] is up, action[1] is down"
        :rtype:
        """
        if random.random() < self.random_thresh:
            if input_vector[1] < input_vector[4]:
                return_val = Direction.UP
            elif input_vector[1] > input_vector[4]:
                return_val = Direction.DOWN
            else:
                return_val = Direction.NOOP
            return return_val
        return random.choice(list(Direction))

    def __str__(self):
        return "RandomHardcodedAi:{}".format(self.random_thresh)


class HardcodedAi:
    def run(self, input_vector):
        """
        :param input_vector: [
            normalised_ball_location_x,
            normalised_ball_location_y,
            normalised_last_ball_location_x,
            normalised_last_ball_location_y,
            normalised_me,
            normalised_enemy
        ]
        :type input_vector:
        :return: action[0] is up, action[1] is down"
        :rtype:
        """
        if input_vector[1] < input_vector[4]:
            return_val = Direction.UP
        elif input_vector[1] > input_vector[4]:
            return_val = Direction.DOWN
        else:
            return_val = Direction.NOOP
        return return_val

    def __str__(self):
        return "HardcodedAi"


class ScoreHardcodedAi:
    def __init__(self):
        self.score_info = {}

    def set_score(self, score_info):
        self.score_info.update(score_info)

    def run(self, input_vector):
        """
        :param input_vector: [
            normalised_ball_location_x,
            normalised_ball_location_y,
            normalised_last_ball_location_x,
            normalised_last_ball_location_y,
            normalised_me,
            normalised_enemy
        ]
        :type input_vector:
        :return: action[0] is up, action[1] is down"
        :rtype:
        """
        return_val = Direction.NOOP
        if self.score_info["score1"] <= self.score_info["score2"]:
            if input_vector[1] < input_vector[4]:
                return_val = Direction.UP
            elif input_vector[1] > input_vector[4]:
                return_val = Direction.DOWN
        return return_val

    def __str__(self):
        return "ScoreHardcodedAi"
