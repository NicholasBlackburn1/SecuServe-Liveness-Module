"""
this class is for detecting and findign and handling computer vision /tensorflow of detecting faces lively 
"""


import cv2
import numpy as np
import sys 
import traceback

class LiveDetection(object):

    def pipelineSetUp(self):
        pass



    def trainModel(self):
        pass


    def runPipeline(self,receiver):


        try:
            #* this llows me to recive frames from my camera over the network
            while True:
                
                msg, frame = receiver.receive()
                image = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)

                
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
        