"""
Simple Debuging Colorizer for the console uwu
"""

import colorama
from datetime import datetime


def info(text):
    print(dmsgLayout("INFO", colorama.Fore.WHITE, text))
    return


def Debug(text): 
    print(dmsgLayout("DEBUG", colorama.Fore.LIGHTBLUE_EX, text))
    return


def Warning(text):
    print(dmsgLayout("WARNING", colorama.Fore.YELLOW, text))
    return


def Error(text):
    print(dmsgLayout("ERROR", colorama.Fore.RED, text))
    return


def PipeLine_Ok(text):
    print(dmsgLayout("OK", colorama.Fore.LIGHTGREEN_EX, text))
    return


def PipeLine_init(text):
    print(dmsgLayout("INIT", colorama.Fore.LIGHTMAGENTA_EX, text))
    return


def PipeLine_Data(text):
    print(dmsgLayout("DATA", colorama.Fore.LIGHTCYAN_EX, text))

    return


def dmsgLayout(type, color, message) -> str:
    return (
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