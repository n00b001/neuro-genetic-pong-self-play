from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    NOOP = 2


FPS = 60

TIMEOUT_THRESH = 4_000

NETWORK_SHAPE = [
    6, 5, 4, len(Direction)
]

BIAS = True

PROBABILITY_OF_MUTATING_A_SINGLE_GENE = 0.9
GAUSSIAN_MUTATION_SIGMA = 0.9
GAUSSIAN_MUTATION_PROBABILITY = 0.9
GAUSSIAN_MUTATION_MEAN = 0

CROSSOVER_BLEND_PROBABILITY = 0.9
CROSSOVER_BLEND_ALPHA = 0.9

GENERATIONS_BEFORE_SAVE = 3
NUMBER_OF_HARD_CODED_RANDOM_AIS = 5
GAMES_TO_PLAY_AGAINST_SELF = 5
GAMES_TO_PLAY = NUMBER_OF_HARD_CODED_RANDOM_AIS + GAMES_TO_PLAY_AGAINST_SELF

# minimum is 10
POPULATION_SIZE = 400

TOURNAMENT_SIZE = POPULATION_SIZE // 4
HALL_OF_FAME_AMOUNT = 10

RENDER = False
WIN_SCORE = 5
TIME_SCALAR = 10000.0

BACKGROUND_COLOUR = (150, 150, 150)
GAME_HEIGHT = 600.0
GAME_WIDTH = 600.0

RIGHT_GUY_COLOUR = (50, 255, 50)
LEFT_GUY_COLOUR = (50, 50, 255)
BALL_COLOUR = (255, 255, 255)
SCORE_COLOUR = (200, 0, 0)

STARTING_POSITION_Y = GAME_HEIGHT / 2.0

PADDLE_EDGE_DISTANCE = 0.0
PADDLE_HEIGHT = 40.0
PADDLE_WIDTH = 10.0
PADDLE_SPEED = 30.0

BALL_SIZE = (10.0, 10.0)
BALL_SPEED = 4.0
BALL_MIN_BOUNCE = 0.1

LEFT_GUY_X = PADDLE_EDGE_DISTANCE
RIGHT_GUY_X = GAME_WIDTH - PADDLE_EDGE_DISTANCE - PADDLE_WIDTH
