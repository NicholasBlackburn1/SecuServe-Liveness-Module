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
import sys
from pipeline.videoStreamSubscriber import VideoStreamSubscriber
from pipeline.LiveDetection import LiveDetection
from pipeline import pipelineStates
from utils import const
import traceback

def main():
    tf.print(consoleLog.Warning("Startig Zmq...."))

    context = zmq.Context(io_threads=2)

    #* recv socket for commands
    recv = context.socket(zmq.SUB)

    recv.setsockopt(zmq.SUBSCRIBE, b"")
    recv.connect(const.zmq_recv)

    #* sender for Socket 
    sender = context.socket(zmq.PUB)
    sender.bind(const.zmq_sender)

    sender.send_string("LIVENESS")
    sender.send_json({'status':"Starting",'alive':False,'time':str(datetime.now)})
    
    #* allows to check the output about 20 times a ms
    poller = zmq.Poller()
    poller.register(recv, zmq.POLLIN)

    print(consoleLog.PipeLine_Ok("Started Zmq..."))


    img_receiver = VideoStreamSubscriber(const.hostname, const.port)

    tf.print(consoleLog.Warning("Connecting to Imgzmq port for frames..."),output_stream=sys.stdout)
    
       # sets pipeline starting state so Fsm has all needed to run
    pipe = pipelineStates.PipeLine()
    pipe.on_event(pipelineStates.States.SETUP_PIPELINE, sender,recv,poller,tf)
    pipe.on_event(pipelineStates.States.TRAIN_MODEL, sender,recv,poller,tf)
    pipe.on_event(pipelineStates.States.RUN_RECONITION, sender,recv,poller,tf)




if __name__ == "__main__":
  main()


    







if __name__ == '__main__':
    main()