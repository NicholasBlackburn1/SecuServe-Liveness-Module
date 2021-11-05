"""
this is the main class for the liveeness dector 
"""

from utils import consoleLog
import zmq
from datetime import datetime
from utility.video_utils import VideoUtils
from face_det.TDDFA import TDDFA
from face_det.FaceBoxes import FaceBoxes
from face_detector import FaceDetector
from utility import main as mn
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
    mn()

    







if __name__ == '__main__':
    main()