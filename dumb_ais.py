class HardcodedAi:
    def run(self, input_vector):
        ret_val = [0, 0]
        if input_vector[1] < input_vector[4]:
            ret_val[0] = 1
        elif input_vector[1] > input_vector[4]:
            ret_val[1] = 1
        return ret_val


class ScoreHardcodedAi:
    def __int__(self):
        self.score_info = {}

    def set_score(self, score_info):
        self.score_info = score_info

    def run(self, input_vector):
        ret_val = [0, 0]
        if self.score_info["score1"] <= self.score_info["score2"]:
            if input_vector[1] < input_vector[4]:
                ret_val[0] = 1
            elif input_vector[1] > input_vector[4]:
                ret_val[1] = 1
        return ret_val
