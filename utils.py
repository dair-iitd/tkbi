import shutil
import os
import sys
import subprocess
from collections import Counter

color_map = {"green": 32, "black": 0, "blue": 34, "red": 31, 'yellow': 33}


def func_load_to_gpu(data, load_to_gpu):
    if load_to_gpu:
        data = data.cuda()
    return data


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=None, fill='*', color="blue"):
    """
    Simple utility to display an updatable progress bar on the terminal.\n
    :param iteration: The numer of iteration completed (% task done)
    :param total: The total number of iterations to be done
    :param prefix: The string to be displayed before the progress bar
    :param suffix: The string to be displayed after the progress bar
    :param decimals: The number of decimal places in the percent complete indicator
    :param length: The length of the bar. If None, the length is calculated as to fill up the screen
    :param fill: The character to fill the bar with
    :param color: Color of the bar
    :return: None
    """
    if (length is None):
        r, _ = shutil.get_terminal_size((120, 80))
        length = max(r - len(prefix) - len(suffix) - 11, 10)
    percent = ("{0:5." + str(decimals) + "f}").format(100.0 * iteration / total)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |\033[1;%dm%s\033[0;0m| %s%% %s' % (prefix, color_map[color], bar, percent, suffix), end='\r')
    if iteration == total:
        print()


def colored_print(color, message):
    """
    Simple utility to print in color
    :param color: The name of color from color_map
    :param message: The message to print in color
    :return: None
    """
    print('\033[1;%dm%s\033[0;0m' % (color_map[color], message))


def duplicate_stdout(filename):
    """
    This function is used to duplicate and redires stdout into a file. This enables a permanent record of the log on
    disk\n
    :param filename: The filename to which stdout should be duplicated
    :return: None
    """
    print("duplicating")
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
    tee = subprocess.Popen(["tee", filename], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def removeElements(lst, k):
    """
    helper for removing elements in list occuring less than k times
    """
    counted = Counter(lst)
    return [el for el in lst if counted[el] >= k]
