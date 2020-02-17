import timeit

from config import *
from utils import get_rect, get_rect_quickly, find_stuff, find_stuff_quickly


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def test_methods(args, meths):
    meth1, meth2 = meths

    output = meth1(*args)
    output2 = meth2(*args)

    if np.any(output != output2):
        # pass
        raise Exception("meth1 out: {}\nmeth2 out: {}".format(output, output2))
    wrapped_1 = wrapper(meth1, *args)
    timer_1 = timeit.Timer(stmt=wrapped_1)
    output_1 = timer_1.autorange()
    wrapped_2 = wrapper(meth2, *args)
    timer_2 = timeit.Timer(stmt=wrapped_2)
    output_2 = timer_2.autorange()
    return [output_1[1], output_2[1]]


def test_get_rects(meths):
    times = []
    _observation = np.load("obs.npy")
    _chopped_observation = _observation[GAME_TOP: GAME_BOTTOM, :]
    all_colours = [BALL_COLOUR, LEFT_GUY_COLOUR, RIGHT_GUY_COLOUR]
    for c in all_colours:
        times.append(test_methods(args=(_chopped_observation, c), meths=meths))
    _chopped_observation = np.zeros_like(_chopped_observation)
    for c in all_colours:
        times.append(test_methods(args=(_chopped_observation, c), meths=meths))
    times = np.array(times)
    avr_1 = sum(times[:, 0]) / len(times)
    avr_2 = sum(times[:, 1]) / len(times)
    return avr_1, avr_2


def test_find_stuff(meths):
    times = []
    _observation = np.load("obs.npy")
    times.append(test_methods(args=(_observation,), meths=meths))
    _observation = np.zeros_like(_observation)
    times.append(test_methods(args=(_observation,), meths=meths))
    times = np.array(times)
    avr_1 = sum(times[:, 0]) / len(times)
    avr_2 = sum(times[:, 1]) / len(times)
    return avr_1, avr_2


if __name__ == '__main__':
    _avr_1, _avr_2 = test_get_rects(meths=(get_rect, get_rect_quickly))
    print("get_rect: {}\nget_rect_quickly: {}\nspeedup: {}%".format(_avr_1, _avr_2, (_avr_1 / _avr_2) * 100.0))
    _avr_1, _avr_2 = test_find_stuff(meths=(find_stuff, find_stuff_quickly))
    print("find_stuff: {}\nfind_stuff_quickly: {}\nspeedup: {}%".format(_avr_1, _avr_2, (_avr_1 / _avr_2) * 100.0))
