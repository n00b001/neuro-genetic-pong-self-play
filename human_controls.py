from pynput import keyboard

from utils import Direction

button_w = 0
button_s = 1
button_up = 2
button_down = 3


class HumanInput:

    def __init__(self):
        self.button_list = [0, 0, 0, 0]
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if type(key) == keyboard.Key:
            if key == keyboard.Key.up:
                self.button_list[button_up] = 1
            elif key == keyboard.Key.down:
                self.button_list[button_down] = 1
        elif type(key) == keyboard.KeyCode:
            if key.char == 'w':
                self.button_list[button_w] = 1
            elif key.char == 's':
                self.button_list[button_s] = 1

    def on_release(self, key):
        if type(key) == keyboard.Key:
            if key == keyboard.Key.up:
                self.button_list[button_up] = 0
            elif key == keyboard.Key.down:
                self.button_list[button_down] = 0
        elif type(key) == keyboard.KeyCode:
            if key.char == 'w':
                self.button_list[button_w] = 0
            elif key.char == 's':
                self.button_list[button_s] = 0


class HumanPlayer1:

    def __init__(self):
        self.input_thing = HumanInput()

    def run(self, input_vector=None):
        if self.input_thing.button_list[button_w]:
            ret_val = Direction.UP
        elif self.input_thing.button_list[button_s]:
            ret_val = Direction.DOWN
        else:
            ret_val = Direction.NOOP
        return ret_val

    def __str__(self):
        return "HP1"


class HumanPlayer2:

    def __init__(self):
        self.input_thing = HumanInput()

    def run(self, input_vector=None):
        if self.input_thing.button_list[button_w]:
            ret_val = Direction.UP
        elif self.input_thing.button_list[button_s]:
            ret_val = Direction.DOWN
        else:
            ret_val = Direction.NOOP
        return ret_val

    def __str__(self):
        return "HP2"
