import contextlib
import math as maths
import random

from consts import *
from utils import rotate, sigmoid

with contextlib.redirect_stdout(None):
    import pygame


class MyPong:
    def __init__(self, render):
        self._running = True
        self.display_surf = None
        self._image_surf = None
        self.left_paddle = pygame.Rect((LEFT_GUY_X, STARTING_POSITION_Y), (PADDLE_WIDTH, PADDLE_HEIGHT))
        self.right_paddle = pygame.Rect((RIGHT_GUY_X, STARTING_POSITION_Y), (PADDLE_WIDTH, PADDLE_HEIGHT))
        self.ball = pygame.Rect((GAME_HEIGHT / 2.0, GAME_WIDTH / 2.0), BALL_SIZE)
        self.system_clock = pygame.time.Clock()
        self.ball_velocity = [0.0, 0.0]
        self.ball_centre_pos = [0.0, 0.0]
        self.ball_centre_pos_previous = [0.0, 0.0]
        self.ball_centre_pos_previous_previous = [0.0, 0.0]
        self.left_paddle_center = [float(LEFT_GUY_X), float(STARTING_POSITION_Y)]
        self.left_paddle_velocity = [0.0, 0.0]
        self.right_paddle_center = [float(RIGHT_GUY_X), float(STARTING_POSITION_Y)]
        self.right_paddle_velocity = [0.0, 0.0]
        self.score = {"score1": 0.0, "score2": 0.0}
        self.score_font = None
        self.debug_font = None
        self.should_render = render
        self.ball_speed_upper = 0.0
        self.total_frames = 0.0
        self.left_frames_since_last_hit = 0.0
        self.right_frames_since_last_hit = 0.0
        self.spin = 0.0
        self.trail = []

    def on_init(self):
        # pygame.init()
        if self.should_render:
            pygame.display.init()
        pygame.font.init()

        if self.should_render:
            self.display_surf = pygame.display.set_mode((int(GAME_WIDTH), int(GAME_HEIGHT)))
        else:
            self.display_surf = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.score_font = pygame.font.Font('font_file.ttf', 100)
        self.debug_font = pygame.font.Font('font_file.ttf', 20)
        return True

    def randomise_ball_vel(self):
        # this offset just means the ball won't hit the paddle on start - it forces the paddle to move

        # 0.79 hits bottom of the screen
        # -0.79 hits top of the screen
        # 0.09 misses the bottom of the paddle
        # -0.09 misses the top of the paddle
        offset = random.uniform(0.09, 0.79)
        # https://stackoverflow.com/questions/6824681/get-a-random-boolean-in-python
        if bool(random.getrandbits(1)):
            # above of below
            offset = -offset
        if bool(random.getrandbits(1)):
            ball_direction = 0.0 + offset
        else:
            ball_direction = maths.pi + offset

        self.ball_velocity = rotate(ball_direction, [BALL_SPEED, 0.0])

    def ball_paddle_redirect(self, location, skill_bonus):
        collision_vet = [
            self.ball_centre_pos[0] - location[0],
            self.ball_centre_pos[1] - location[1]
        ]
        collision_angle = maths.atan2(collision_vet[1], collision_vet[0])

        if abs((maths.pi / 2.0) - abs(collision_angle)) < BALL_MIN_BOUNCE:
            if collision_angle < 0:
                collision_angle = collision_angle + BALL_MIN_BOUNCE
            else:
                collision_angle = collision_angle - BALL_MIN_BOUNCE

        # the 1.6 here has been tested - it is the fastest the ball can go before it goes through the paddles
        # use 1.4 to be safe
        ball_speed = min(float(BALL_SPEED + (self.ball_speed_upper + skill_bonus)), BALL_SIZE[0] * 1.4)
        self.ball_velocity = rotate(collision_angle, [ball_speed, 0.0])

    def get_velocity(self, control):
        return_val = [0.0, 0.0]
        if control == Direction.UP:
            return_val = [0, -PADDLE_SPEED]
        elif control == Direction.DOWN:
            return_val = [0, PADDLE_SPEED]
        return return_val

    def limit_paddles(self, location):
        top = location[1] - (PADDLE_HEIGHT / 2.0)
        bottom = location[1] + (PADDLE_HEIGHT / 2.0)
        if top < 0:
            location[1] = (PADDLE_HEIGHT / 2.0)
        elif bottom > GAME_HEIGHT:
            location[1] = GAME_HEIGHT - (PADDLE_HEIGHT / 2.0)
        return location

    def collide_checks(self):
        collide = False
        if self.ball_centre_pos[1] < BALL_SIZE[1] / 2.0:
            self.spin *= 0.5
            self.ball_velocity[1] = abs(self.ball_velocity[1])
            self.ball_centre_pos[1] = (BALL_SIZE[1] / 2.0)
            collide = True
        elif self.ball_centre_pos[1] > GAME_HEIGHT - (BALL_SIZE[1] / 2.0):
            self.spin *= 0.5
            self.ball_velocity[1] = -abs(self.ball_velocity[1])
            self.ball_centre_pos[1] = GAME_HEIGHT - (BALL_SIZE[1] / 2.0)
            collide = True

        if self.ball.colliderect(self.left_paddle):
            self.left_frames_since_last_hit = 0.0
            skill_bonus = 0.0
            if self.ball_centre_pos[0] - (BALL_SIZE[0] / 2.0) < self.left_paddle_center[0]:
                skill_bonus = 10.0
            self.ball_centre_pos[0] = self.left_paddle_center[0] + (BALL_SIZE[0] / 2.0) + (PADDLE_WIDTH / 2.0)
            self.ball_paddle_redirect(self.left_paddle_center, skill_bonus)
            self.spin = self.calculate_spin(self.left_paddle_velocity[1])
            # self.ball_velocity[1] -
            self.score["score1"] += PADDLE_HIT_SCORE
            self.score["score2"] -= PADDLE_HIT_SCORE
            collide = True
        elif self.ball.colliderect(self.right_paddle):
            self.right_frames_since_last_hit = 0.0
            skill_bonus = 0.0
            if self.ball_centre_pos[0] + (BALL_SIZE[0] / 2.0) > self.right_paddle_center[0]:
                skill_bonus = 10.0
            self.ball_centre_pos[0] = self.right_paddle_center[0] - (BALL_SIZE[0] / 2.0) - (PADDLE_WIDTH / 2.0)
            self.ball_paddle_redirect(self.right_paddle_center, skill_bonus)
            self.spin = self.calculate_spin(self.right_paddle_velocity[1])
            self.score["score2"] += PADDLE_HIT_SCORE
            self.score["score1"] -= PADDLE_HIT_SCORE
            collide = True

        self.left_paddle_center = self.limit_paddles(self.left_paddle_center)
        self.right_paddle_center = self.limit_paddles(self.right_paddle_center)

        if collide:
            # todo: this speeds up the ball
            self.ball_speed_upper += BALL_SPEED_UPPER

            # todo
            # This is to introduce a little randomness to reduce the chance of looping forever...
            #   not sure if it's a good idea
            #   the "4" her has been tested
            # self.ball_velocity = [
            #     self.ball_velocity[0] + ((random.random() - 0.5) * 4.0),
            #     self.ball_velocity[1] + ((random.random() - 0.5) * 4.0)
            # ]

    def calculate_spin(self, vel):
        spin = (sigmoid(vel) - 0.5) * (maths.pi / 72.0)
        thresh = 0.006
        if spin > thresh:
            spin = thresh
        elif spin < -thresh:
            spin = -thresh
        return spin

    def score_logic(self):
        scored = False
        if self.ball_centre_pos[0] < 0.0:
            scored = True
            self.score["score2"] += POINT_SCORE
            self.score["score1"] -= POINT_SCORE
        elif self.ball_centre_pos[0] > GAME_WIDTH:
            scored = True
            self.score["score1"] += POINT_SCORE
            self.score["score2"] -= POINT_SCORE
        if scored:
            self.restart_ball()
            self.restart_paddles()
            self.ball_speed_upper = 0.0

    def restart_ball(self):
        self.trail = []
        self.spin = 0.0
        self.ball_centre_pos[0] = GAME_WIDTH / 2.0
        self.ball_centre_pos[1] = GAME_HEIGHT / 2.0
        self.ball_centre_pos_previous = self.ball_centre_pos
        self.ball_centre_pos_previous_previous = self.ball_centre_pos_previous
        self.randomise_ball_vel()

    def restart_paddles(self):
        self.left_paddle_center[1] = GAME_HEIGHT / 2.0
        self.right_paddle_center[1] = GAME_HEIGHT / 2.0

    def reset(self):
        self._running = self.on_init()
        self.restart_ball()
        self.restart_paddles()
        self.ball_speed_upper = 0.0
        self.total_frames = 0.0
        self.left_frames_since_last_hit = 0.0
        self.right_frames_since_last_hit = 0.0
        self.score = {"score1": 0, "score2": 0}

    def render(self, left_class=None, right_class=None):
        self.display_surf.fill(BACKGROUND_COLOUR)

        score_string = self.get_score_string()
        self.draw_string(
            score_string,
            GAME_WIDTH / 2.0,
            50,
            self.score_font
        )

        if left_class is not None:
            left_debug_string = "{}".format(left_class)
            self.draw_string(
                left_debug_string,
                GAME_WIDTH / 4.0,
                GAME_HEIGHT - 10,
                self.debug_font
            )
        if right_class is not None:
            right_debug_string = "{}".format(right_class)
            self.draw_string(
                right_debug_string,
                GAME_WIDTH - (GAME_WIDTH / 4.0),
                GAME_HEIGHT - 10,
                self.debug_font
            )
        # self.draw_string(
        #     "{}".format(self.spin),
        #     GAME_WIDTH / 2.0,
        #     GAME_HEIGHT / 2.0,
        #     self.debug_font
        # )

        pygame.draw.rect(self.display_surf, LEFT_GUY_COLOUR, self.left_paddle)
        pygame.draw.rect(self.display_surf, RIGHT_GUY_COLOUR, self.right_paddle)

        if len(self.trail) > 2:
            pygame.draw.lines(self.display_surf, BALL_COLOUR, False, self.trail, 2)
        pygame.draw.rect(self.display_surf, BALL_COLOUR, self.ball)
        pygame.display.flip()

    def draw_string(self, string, x, y, font):
        text = font.render(string, True, SCORE_COLOUR, BACKGROUND_COLOUR)
        rectangle = text.get_rect()
        rectangle.center = (x, y)
        self.display_surf.blit(text, rectangle)

    def get_score_string(self):
        score_string = "{:.2f}:{:.2f}".format(self.score["score1"], self.score["score2"])
        return score_string

    def on_cleanup(self):
        pygame.quit()

    """
    should return:
    observation, reward, is_done, score_info 
    """

    def step(self, control):
        if self.render:
            self.trail.append(self.ball_centre_pos)
            if len(self.trail) > 100:
                self.trail.pop(0)

        self.total_frames += 1.0
        self.left_frames_since_last_hit += 1.0
        self.right_frames_since_last_hit += 1.0
        self.spin = self.spin * 0.999

        self.score = self.multiply_score(SCORE_DECAY)
        self.left_paddle_velocity = self.get_velocity(control["player1"])
        self.right_paddle_velocity = self.get_velocity(control["player2"])

        self.collide_checks()
        self.update()
        self.score_logic()

        self.left_paddle.center = self.left_paddle_center
        self.right_paddle.center = self.right_paddle_center
        self.ball.center = self.ball_centre_pos

        observation = [
            [self.ball_centre_pos[1], self.ball_centre_pos[0]],
            [self.ball_centre_pos_previous[1], self.ball_centre_pos_previous[0]],
            [self.ball_centre_pos_previous_previous[1], self.ball_centre_pos_previous_previous[0]],
            [self.left_paddle_center[1], self.left_paddle_center[0]],
            [self.right_paddle_center[1], self.right_paddle_center[0]]
        ]

        try:
            pygame.event.get()  # we must do this to stop it freezing on windows :(
        except:
            pass

        if self.total_frames > TOTAL_TIMEOUT_THRESH:
            self._running = False
        if self.left_frames_since_last_hit > LAST_HIT_TIMEOUT_THRESH:
            self._running = False
        if self.right_frames_since_last_hit > LAST_HIT_TIMEOUT_THRESH:
            self._running = False
        return observation, 0, self._running, self.score

    def update(self):
        self.ball_velocity = rotate(self.spin, self.ball_velocity)

        ball_pos = [
            self.ball_centre_pos[i] + (self.ball_velocity[i] * TIME_STEP)
            for i in range(len(self.ball_velocity))
        ]
        self.ball_centre_pos_previous_previous = self.ball_centre_pos_previous
        self.ball_centre_pos_previous = self.ball_centre_pos
        self.ball_centre_pos = ball_pos

        left_paddle_pos = [
            self.left_paddle_center[i] + (self.left_paddle_velocity[i] * TIME_STEP)
            for i in range(len(self.left_paddle_velocity))
        ]
        self.left_paddle_center = left_paddle_pos

        right_paddle_pos = [
            self.right_paddle_center[i] + (self.right_paddle_velocity[i] * TIME_STEP)
            for i in range(len(self.right_paddle_velocity))
        ]
        self.right_paddle_center = right_paddle_pos

    def multiply_score(self, number):
        return {k: v * number for k, v in self.score.items()}
