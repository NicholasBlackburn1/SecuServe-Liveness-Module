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
import traceback

def main():
    tf.print(consoleLog.Warning("Startig Zmq...."))

    context = zmq.Context()
    sender = context.socket(zmq.PUB)
    sender.bind("tcp://" + "127.0.0.1:5000")

    sender.send_string("LIVENESS")
    sender.send_json({'status':"Starting",'alive':False,'time':str(datetime.now)})
    
    print(consoleLog.PipeLine_Ok("Started Zmq..."))


    hostname = "127.0.0.1"  # Use to receive from localhost
    port = 5555
    receiver = VideoStreamSubscriber(hostname, port)

    tf.print(consoleLog.Warning("Connecting to Imgzmq port for frames..."),output_stream=sys.stdout)
    
    LiveDetection.runPipeline(LiveDetection,receiver)
    


if __name__ == "__main__":
  main()


    







if __name__ == '__main__':
    main()