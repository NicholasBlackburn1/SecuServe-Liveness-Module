"""
this is the main class for the liveeness dector 
TODO: need to actually get the liveness dector working 
"""

from utils import consoleLog
import zmq
from datetime import datetime
import cv2
import yaml
import numpy as np
import tensorflow as tf


def main():
    consoleLog.Warning("Starting Zmq....")

    context = zmq.Context()
    sender = context.socket(zmq.PUB)
    sender.bind("tcp://" + "127.0.0.1:5000")

    sender.send_string("LIVENESS")
    sender.send_json({'status':"Starting",'alive':False,'time':str(datetime.now)})
    
    consoleLog.PipeLine_Ok("Started Zmq...")


    







if __name__ == '__main__':
    main()