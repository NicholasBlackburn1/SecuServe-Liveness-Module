"""
this class is for detecting and findign and handling computer vision /tensorflow of detecting faces lively 
"""


import cv2
import numpy as np
import sys 
import traceback
from utils import consoleLog

class LiveDetection(object):


    # Model path
    FAS_MODEL_PATH = "model/anti_spoofing.h5"
    FACE_THRESHOLD = 0.9
    BLUR_THRESHOLD = 350


    #* sets up pipeline 
    def pipelineSetUp(self,tf,sender):
        
        consoleLog.Warning("Staring to set up gpu in tensorflow")

        # Allow GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:

                    tf.config.experimental.set_memory_growth(gpu, True)
                    consoleLog.PipeLine_Ok("set Tensorflow to use all mem gpu")

            except RuntimeError as e:

                consoleLog.Error(e)

       

        

    
    #* this is where the tensorflow will run our liveness detection 
    def runPipeline(self,img_receiver,sender,recv,poller):


        try:
            #* this llows me to recive frames from my camera over the network
            while True:
                
                msg, frame = img_receiver.receive()
                image = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)

                cv2.imshow("wo",image)
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

    #* does a blink calculation on my frames
    def detectBlinking(self):
        pass

    #* sends status updates from liveness detection 
    def sendLifeStatus(self,sender):
        pass
        


    def sendProgramStatus(self, sender):
