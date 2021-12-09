"""
Simple Debuging Colorizer for the console uwu
"""

import colorama
from datetime import datetime
import tensorflow as tf

def info(text):
    tf.get_logger().info(dmsgLayout("INFO", colorama.Fore.WHITE, text))
    return


def Debug(text): 
    tf.get_logger().debug(dmsgLayout("DEBUG", colorama.Fore.LIGHTBLUE_EX, text))
    return


def Warning(text):
    tf.get_logger().warning(dmsgLayout("WARNING", colorama.Fore.YELLOW, text))
    return


def Error(text):
    tf.get_logger().error(dmsgLayout("ERROR", colorama.Fore.RED, text))
    return


def PipeLine_Ok(text):
    tf.get_logger().info(dmsgLayout("OK", colorama.Fore.LIGHTGREEN_EX, text))
    return


def PipeLine_init(text):
    tf.get_logger().info(dmsgLayout("INIT", colorama.Fore.LIGHTMAGENTA_EX, text))
    return


def PipeLine_Data(text):
    tf.get_logger().info(dmsgLayout("DATA", colorama.Fore.LIGHTCYAN_EX, text))

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