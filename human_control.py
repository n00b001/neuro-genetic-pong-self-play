from pynput import keyboard

button_w = 0
button_s = 1
button_list = [0, 0]


def on_press(key):
    try:
        if key.char == 'w':
            button_list[button_w] = 1
        elif key.char == 's':
            button_list[button_s] = 1
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


def on_release(key):
    if key.char == 'w':
        button_list[button_w] = 0
    elif key.char == 's':
        button_list[button_s] = 0


class HumanInput:

    def __init__(self):
        # ...or, in a non-blocking fashion:
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        self.listener.start()
        pass

    def run(self, input_vector=None):
        return [button_list[button_w], button_list[button_s]]
