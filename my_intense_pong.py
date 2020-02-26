import contextlib
import math as maths
import random

from config import *
from utils import Direction

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
        self.score = {"score1": 0.0, "score2": 0.0}
        self.score_font = None
        self.debug_font = None
        self.should_render = render

    def on_init(self):
        pygame.init()
        if self.should_render:
            self.display_surf = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        else:
            self.display_surf = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.score_font = pygame.font.Font('font_file.ttf', 100)
        self.debug_font = pygame.font.Font('font_file.ttf', 20)
        self.randomise_ball_vel()
        return True

    def randomise_ball_vel(self):
        # https://stackoverflow.com/questions/6824681/get-a-random-boolean-in-python
        if bool(random.getrandbits(1)):
            ball_direction = 0.0
        else:
            ball_direction = maths.pi
        self.ball_velocity = [
            maths.cos(float(ball_direction)) * float(BALL_SPEED),
            maths.sin(float(ball_direction)) * float(BALL_SPEED)
        ]

    def ball_paddle_redirect(self, paddle: pygame.Rect):
        collision_vet = [self.ball.center[0] - paddle.center[0], self.ball.center[1] - paddle.center[1]]
        collision_angle = maths.atan2(collision_vet[1], collision_vet[0])

        if abs((maths.pi / 2.0) - abs(collision_angle)) < BALL_MIN_BOUNCE:
            if collision_angle < 0:
                collision_angle = collision_angle + BALL_MIN_BOUNCE
            else:
                collision_angle = collision_angle - BALL_MIN_BOUNCE

        self.ball_velocity = [
            maths.cos(float(collision_angle)) * float(BALL_SPEED),
            maths.sin(float(collision_angle)) * float(BALL_SPEED)
        ]

    def move_paddle(self, paddle: pygame.Rect, control):
        if control == Direction.UP:
            paddle = paddle.move(0, -PADDLE_SPEED)
        elif control == Direction.DOWN:
            paddle = paddle.move(0, PADDLE_SPEED)
        return paddle

    def limit_paddles(self, paddle: pygame.Rect):
        if paddle.top < 0:
            paddle.top = 0
        elif paddle.bottom > GAME_HEIGHT:
            paddle.bottom = GAME_HEIGHT
        return paddle

    def bounce_ball(self):
        if self.ball.top < 0:
            self.ball_velocity[1] = abs(self.ball_velocity[1])
        elif self.ball.bottom > GAME_HEIGHT:
            self.ball_velocity[1] = -abs(self.ball_velocity[1])

        if self.ball.colliderect(self.left_paddle):
            self.ball_paddle_redirect(self.left_paddle)
        elif self.ball.colliderect(self.right_paddle):
            self.ball_paddle_redirect(self.right_paddle)

        if self.ball.left < 0:
            self.score["score2"] += 1
        elif self.ball.right > GAME_WIDTH:
            self.score["score1"] += 1
        if self.ball.left < 0 or self.ball.right > GAME_WIDTH:
            self.restart_ball()
            self.restart_paddles()

    def restart_ball(self):
        self.ball.centerx = GAME_WIDTH / 2.0
        self.ball.centery = GAME_HEIGHT / 2.0
        self.randomise_ball_vel()

    def restart_paddles(self):
        self.left_paddle.centery = GAME_HEIGHT / 2.0
        self.right_paddle.centery = GAME_HEIGHT / 2.0

    def reset(self):
        self._running = self.on_init()
        self.restart_ball()
        self.restart_paddles()
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

        pygame.draw.rect(self.display_surf, LEFT_GUY_COLOUR, self.left_paddle)
        pygame.draw.rect(self.display_surf, RIGHT_GUY_COLOUR, self.right_paddle)
        pygame.draw.rect(self.display_surf, BALL_COLOUR, self.ball)
        pygame.display.flip()

    def draw_string(self, string, x, y, font):
        text = font.render(string, True, SCORE_COLOUR, BACKGROUND_COLOUR)
        rectangle = text.get_rect()
        rectangle.center = (x, y)
        self.display_surf.blit(text, rectangle)

    def get_score_string(self):
        score_string = "{}:{}".format(self.score["score1"], self.score["score2"])
        return score_string

    def on_cleanup(self):
        pygame.quit()

    """
    should return:
    observation, reward, is_done, score_info 
    """

    def step(self, control):
        self.left_paddle = self.move_paddle(self.left_paddle, control["player1"])
        self.right_paddle = self.move_paddle(self.right_paddle, control["player2"])

        self.left_paddle = self.limit_paddles(self.left_paddle)
        self.right_paddle = self.limit_paddles(self.right_paddle)
        self.ball = self.ball.move(self.ball_velocity)
        self.bounce_ball()

        observation = [
            [self.ball.centery, self.ball.centerx],
            [self.left_paddle.centery, self.left_paddle.centerx],
            [self.right_paddle.centery, self.right_paddle.centerx]
        ]

        pygame.event.get()  # we must do this to stop it freezing on windows :(
        return observation, 0, False, self.score

    def get_debug_string(self, left_class, right_class):
        debug_string = ""
        if left_class is not None:
            debug_string += "{}\n".format(left_class)
        if right_class is not None:
            debug_string += "{}\n".format(right_class)
        debug_string = debug_string[:-1]
        return debug_string
