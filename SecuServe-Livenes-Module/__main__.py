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
from utils import const
import traceback

def main():
    tf.print(consoleLog.Warning("Startig Zmq...."))

    context = zmq.Context(io_threads=2)

    #* recv socket for commands
    recv = context.socket(zmq.SUB)
    recv.connect(const.zmq_recv)

    #* sender for Socket info
    sender = context.socket(zmq.PUB)
    sender.bind(const.zmq_sender)

    sender.send_string("LIVENESS")
    sender.send_json({'status':"Starting",'alive':False,'time':str(datetime.now)})
    
    #* allows to check the output about 20 times a ms
    poller = zmq.Poller()
    poller.register(recv, zmq.POLLIN)

    print(consoleLog.PipeLine_Ok("Started Zmq..."))


    receiver = VideoStreamSubscriber(const.hostname, const.port)

    tf.print(consoleLog.Warning("Connecting to Imgzmq port for frames..."),output_stream=sys.stdout)
    
    LiveDetection.runPipeline(LiveDetection,receiver)
    


if __name__ == "__main__":
  main()


    







if __name__ == '__main__':
    main()