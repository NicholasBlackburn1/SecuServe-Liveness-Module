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

import sys
from pipeline.videoStreamSubscriber import VideoStreamSubscriber
from pipeline.LiveDetection import LiveDetection
from pipeline import pipelineStates
from utils import const
import traceback


def main():
    
    tf = 0
    consoleLog.Warning("Startig Zmq....")

    context = zmq.Context(io_threads=4)

    #* sender for Socket 
    sender = context.socket(zmq.PUSH)
    sender.bind(const.zmq_sender)

    consoleLog.PipeLine_Ok("Started Zmq...")


    img_receiver = VideoStreamSubscriber(const.hostname, const.port)

    consoleLog.Warning("Starting tp run Pipeline")
    
       # sets pipeline starting state so Fsm has all needed to run
    pipe = pipelineStates.PipeLine()
    pipe.on_event(pipelineStates.States.SETUP_PIPELINE, sender,tf,img_receiver)
    pipe.on_event(pipelineStates.States.TRAIN_MODEL, sender,tf,img_receiver)
    pipe.on_event(pipelineStates.States.RUN_RECONITION, sender,tf,img_receiver)




if __name__ == "__main__":
  main()


    




