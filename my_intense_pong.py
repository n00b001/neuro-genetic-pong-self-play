import pygame
from pygame.locals import *
from config import *
from human_controls import HumanPlayer1, HumanPlayer2
from hardcoded_ai import HardcodedAi
import random
import math as maths


class App:
    def __init__(self, player_one, player_two):
        self._running = True
        self.display_surf = None
        self._image_surf = None
        self.left_paddle = pygame.Rect((LEFT_GUY_X, STARTING_POSITION_Y), (PADDLE_WIDTH, PADDLE_HEIGHT))
        self.right_paddle = pygame.Rect((RIGHT_GUY_X, STARTING_POSITION_Y), (PADDLE_WIDTH, PADDLE_HEIGHT))
        self.ball = pygame.Rect((GAME_HEIGHT/2, GAME_WIDTH/2), BALL_SIZE)
        self.player_one = player_one
        self.player_two = player_two
        self.system_clock = pygame.time.Clock()
        self.ball_velocity = [0, 0]
        self.score = [0, 0]
        self.text_obj = None

    def on_init(self):
        pygame.init()
        self.display_surf = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        self.text_obj = pygame.font.Font('font_file.ttf', 100)
        self.randomise_ball_vel()
        return True

    def randomise_ball_vel(self):
        ball_direction = (random.random() * (maths.pi/2)) - maths.pi/4
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

        self.ball_velocity = [maths.ceil(maths.cos(collision_angle) * BALL_SPEED), maths.ceil(maths.sin(collision_angle) * BALL_SPEED)]

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    # ballx = 0, bally = 1, lastballx = 2, lastbally = 3, mey = 4, enemyy = 5

    def on_loop(self):
        control = self.player_one.run()
        self.left_paddle = self.move_paddle(self.left_paddle, control)

        control = self.player_two.run([self.ball.centery, 0, 0, 0, self.right_paddle.centery, 0])
        self.right_paddle = self.move_paddle(self.right_paddle, control)

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
            self.score[1] = self.score[1] + 1
            self.restart_ball()

        if self.ball.right > GAME_WIDTH:
            self.score[0] = self.score[0] + 1
            self.restart_ball()

    def restart_ball(self):
        self.ball.centerx = GAME_WIDTH/2
        self.ball.centery = GAME_HEIGHT/2
        self.randomise_ball_vel()

    def on_render(self):
        self.display_surf.fill(BACKGROUND_COLOUR)
        score_string = "".join((str(self.score[0]), ":", str(self.score[1])))
        score_text = self.text_obj.render(score_string, True, SCORE_COLOUR, BACKGROUND_COLOUR)
        score_rectangle = score_text.get_rect()
        score_rectangle.center = (GAME_WIDTH//2, GAME_HEIGHT//2)
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
            self.on_render()
        self.on_cleanup()

    def step(self):
        self.on_loop()


if __name__ == "__main__":
    player_one = HumanPlayer1()
    player_two = HardcodedAi()
    theApp = App(player_one=player_one, player_two=player_two)
    theApp.on_execute()
