import math as maths
# from hardcoded_ai import HardcodedAi
import random

import pygame
from pygame.locals import *

from config import RIGHT_ACTION_START, RIGHT_ACTION_END, LEFT_ACTION_END
from dumb_ais import HardcodedAi
from game_config import *
from human_controls import HumanPlayer1
from utils import inference, get_actions


class MyPong:
    def __init__(self, player_one, player_two):
        self._running = True
        self.display_surf = None
        self._image_surf = None
        self.left_paddle = pygame.Rect((LEFT_GUY_X, STARTING_POSITION_Y), (PADDLE_WIDTH, PADDLE_HEIGHT))
        self.right_paddle = pygame.Rect((RIGHT_GUY_X, STARTING_POSITION_Y), (PADDLE_WIDTH, PADDLE_HEIGHT))
        self.ball = pygame.Rect((GAME_HEIGHT / 2, GAME_WIDTH / 2), BALL_SIZE)
        self.player_one = player_one
        self.player_two = player_two
        self.system_clock = pygame.time.Clock()
        self.ball_velocity = [0, 0]
        self.score = {"score1": 0, "score2": 0}
        self.text_obj = None
        self.is_done = False

    def on_init(self):
        pygame.init()
        self.display_surf = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        self.text_obj = pygame.font.Font('font_file.ttf', 100)
        self.randomise_ball_vel()
        return True

    def randomise_ball_vel(self):
        ball_direction = (random.random() * (maths.pi / 2)) - maths.pi / 4
        if random.randint(0, 1) == 1:
            ball_direction = ball_direction + maths.pi
        self.ball_velocity = [maths.cos(ball_direction) * BALL_SPEED, maths.sin(ball_direction) * BALL_SPEED]

    def ball_paddle_redirect(self, paddle: pygame.Rect):
        collision_vet = [self.ball.center[0] - paddle.center[0], self.ball.center[1] - paddle.center[1]]
        collision_angle = maths.atan2(collision_vet[1], collision_vet[0])

        if abs((maths.pi / 2) - abs(collision_angle)) < BALL_MIN_BOUNCE:
            if collision_angle < 0:
                collision_angle = collision_angle + BALL_MIN_BOUNCE
            else:
                collision_angle = collision_angle - BALL_MIN_BOUNCE

        self.ball_velocity = [maths.ceil(maths.cos(collision_angle) * BALL_SPEED),
                              maths.ceil(maths.sin(collision_angle) * BALL_SPEED)]

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    # ballx = 0, bally = 1, lastballx = 2, lastbally = 3, mey = 4, enemyy = 5

    def on_loop(self):
        left_action, right_action = get_actions(
            [self.ball.centery, self.ball.centerx],
            [0, 0],
            [self.left_paddle.centery, self.left_paddle.centerx],
            self.player_one,
            [self.right_paddle.centery, self.right_paddle.centerx],
            self.player_two
        )

        self.left_paddle = self.move_paddle(self.left_paddle, left_action)
        self.right_paddle = self.move_paddle(self.right_paddle, right_action)

        self.left_paddle = self.limit_paddles(self.left_paddle)
        self.right_paddle = self.limit_paddles(self.right_paddle)
        self.ball = self.ball.move(self.ball_velocity)
        self.bounce_ball()

    def move_paddle(self, paddle: pygame.Rect, control):
        if control[0] == 1:
            paddle = paddle.move(0, -PADDLE_SPEED)
        elif control[1] == 1:
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
            self.restart_ball()
            self.is_done = True
        elif self.ball.right > GAME_WIDTH:
            self.score["score1"] += 1
            self.restart_ball()
            self.is_done = True
        else:
            self.is_done = False

    def restart_ball(self):
        self.ball.centerx = GAME_WIDTH / 2
        self.ball.centery = GAME_HEIGHT / 2
        self.randomise_ball_vel()

    def reset(self):
        self._running = self.on_init()

    def render(self):
        self.display_surf.fill(BACKGROUND_COLOUR)
        score_string = ":".join([str(x) for x in list(self.score.values())])
        score_text = self.text_obj.render(score_string, True, SCORE_COLOUR, BACKGROUND_COLOUR)
        score_rectangle = score_text.get_rect()
        score_rectangle.center = (GAME_WIDTH // 2, GAME_HEIGHT // 2)
        self.display_surf.blit(score_text, score_rectangle)
        pygame.draw.rect(self.display_surf, LEFT_GUY_COLOUR, self.left_paddle)
        pygame.draw.rect(self.display_surf, RIGHT_GUY_COLOUR, self.right_paddle)
        pygame.draw.rect(self.display_surf, BALL_COLOUR, self.ball)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self._running = self.on_init()

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.system_clock.tick(FRAME_RATE)
            self.on_loop()
            self.render()
        self.on_cleanup()

    """
    should return:
    observation, reward, is_done, score_info 
    """

    def step(self, control):
        self.left_paddle = self.move_paddle(self.left_paddle, control["player1"].value)
        self.right_paddle = self.move_paddle(self.right_paddle, control["player2"].value)

        self.left_paddle = self.limit_paddles(self.left_paddle)
        self.right_paddle = self.limit_paddles(self.right_paddle)
        self.ball = self.ball.move(self.ball_velocity)
        self.bounce_ball()

        observation = [
            [self.ball.x, self.ball.y],
            [self.left_paddle.x, self.left_paddle.y],
            [self.right_paddle.x, self.right_paddle.y]
        ]

        return observation, 0, self.is_done, self.score


if __name__ == "__main__":
    player_one = HumanPlayer1()
    player_two = HardcodedAi()
    theApp = MyPong(player_one=player_one, player_two=player_two)
    theApp.on_execute()
