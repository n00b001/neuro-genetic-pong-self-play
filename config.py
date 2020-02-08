import numpy as np

BG_COLOUR = (144, 72, 17)
BALL_COLOUR = (236, 236, 236)
LEFT_GUY_COLOUR = (213, 130, 74)
RIGHT_GUY_COLOUR = (92, 186, 92)

SCALE_FACTOR = 2
GAME_BOTTOM = 194
GAME_TOP = 34
SCALED_PADDLE_HEIGHT = 16.0 / SCALE_FACTOR
FPS = 600
GAME_PLAYABLE_HEIGHT = GAME_BOTTOM - GAME_TOP
GAME_WIDTH = 160

RIGHT_ACTION_START = 4
RIGHT_ACTION_END = 6
LEFT_ACTION_END = 8
LEFT_PLAYER_START_BUTTON = -1
RIGHT_PLAYER_START_BUTTON = 0

BLANK_ACTION = np.zeros(shape=(16,), dtype=np.int)
BLANK_ACTION[LEFT_PLAYER_START_BUTTON] = 1
BLANK_ACTION[RIGHT_PLAYER_START_BUTTON] = 1

N_CLASSES = 2
ALL_ACTIONS = np.eye(N_CLASSES, dtype=np.int)

TIMEOUT_THRESH = 2_000
# GAMES_TO_PLAY = 1
GAMES_TO_PLAY = 2

GENE_SIZE = 24
IND_PB = 0.2
SIGMA = 1
MU = 0

CX_PB = 0.5
MUT_PB = 0.2
N_GENS = 10

# POPULATION_SIZE = 20
POPULATION_SIZE = 8
HALL_OF_FAME_AMOUNT = 3

RENDER = False
