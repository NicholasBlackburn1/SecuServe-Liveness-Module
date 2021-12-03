"""
this class is for detecting and findign and handling computer vision /tensorflow of detecting faces lively 
"""


import cv2
import numpy as np
import sys 
import traceback
from realtime.utility.video_utils import VideoUtils

class LiveDetection(object):


    # Model path
    FAS_MODEL_PATH = "model/anti_spoofing.h5"
    FACE_THRESHOLD = 0.9
    BLUR_THRESHOLD = 350


    #* sets up pipeline 
    def pipelineSetUp(self,tf):
                
        # Allow GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    
    #* this is where the tensorflow will run our liveness detection 
    def runPipeline(self,receiver):


        try:
            #* this llows me to recive frames from my camera over the network
            while True:
                
                msg, frame = receiver.receive()
                image = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)

        except (KeyboardInterrupt, SystemExit):
            print('Exit due to keyboard interrupt')

        except Exception as ex:
            print('Python error with no Exception handler:')
            print('Traceback error:', ex)
            traceback.print_exc()
            
        finally:
            receiver.close()
            sys.exit()

    #* does a blink calculation on my frames
    def detectBlinking(self):
        pass

    #* sends status updates from liveness detection 
    def sendLifeStatus(self,sender):
        pass
        