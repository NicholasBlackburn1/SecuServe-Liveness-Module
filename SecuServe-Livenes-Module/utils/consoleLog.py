"""
Simple Debuging Colorizer for the console uwu
"""

import colorama
from datetime import datetime

def info(text):
    dmsgLayout("INFO", colorama.Fore.WHITE, text)
    return


def Debug(text):
    dmsgLayout("DEBUG", colorama.Fore.LIGHTBLUE_EX, text)
    return


def Warning(text):
    dmsgLayout("WARNING", colorama.Fore.YELLOW, text)
    return


def Error(text):
    dmsgLayout("ERROR", colorama.Fore.RED, text)
    return


def PipeLine_Ok(text):
    dmsgLayout("OK", colorama.Fore.LIGHTGREEN_EX, text)
    return


def PipeLine_init(text):
    dmsgLayout("INIT", colorama.Fore.LIGHTMAGENTA_EX, text)
    return


def PipeLine_Data(text):
    dmsgLayout("DATA", colorama.Fore.LIGHTCYAN_EX, text)

    return


def dmsgLayout(type, color, message):
    print(
        colorama.Fore.GREEN
        + "["
        + str(datetime.now())
        + "]"
        + " "
        + color
        + str(type)
        + ":"
        + " "
        + colorama.Fore.WHITE
        + str(message)+"\n"
    )