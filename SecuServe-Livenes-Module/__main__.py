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
import traceback

def main():
    consoleLog.Warning("Starting Zmq....")

    context = zmq.Context()
    sender = context.socket(zmq.PUB)
    sender.bind("tcp://" + "127.0.0.1:5000")

    sender.send_string("LIVENESS")
    sender.send_json({'status':"Starting",'alive':False,'time':str(datetime.now)})
    
    consoleLog.PipeLine_Ok("Started Zmq...")


if __name__ == "__main__":
    # Receive from broadcast
    # There are 2 hostname styles; comment out the one you don't need
    hostname = "127.0.0.1"  # Use to receive from localhost
    # hostname = "192.168.86.38"  # Use to receive from other computer
    port = 5555
    receiver = VideoStreamSubscriber(hostname, port)

    try:
        while True:
            msg, frame = receiver.receive()
            image = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)

            # Due to the IO thread constantly fetching images, we can do any amount
            # of processing here and the next call to receive() will still give us
            # the most recent frame (more or less realtime behaviour)

            # Uncomment this statement to simulate processing load
            # limit_to_2_fps()   # Comment this statement out to run full speeed

            cv2.imshow("Pub Sub Receive", image)
            cv2.waitKey(1)
    except (KeyboardInterrupt, SystemExit):
        print('Exit due to keyboard interrupt')
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        receiver.close()
        sys.exit()


    







if __name__ == '__main__':
    main()