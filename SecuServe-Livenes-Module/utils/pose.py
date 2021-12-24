""" 
this is a pose detector in tensorflow so i can detect blinking and a skeletion tracking 


"""

import cv2
import mediapipe as mp
import time
from utils import consoleLog
import tensorflow as tf
class PoseDetector:

    def __init__(self, mode = False, upBody = True, smooth=False, detectionCon = True, trackCon = 0.3):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus, 'GPU')


    def findPose(self, img, draw=False):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
       
        if self.results.pose_landmarks:
            consoleLog.Warning("Sees Person with A Body uwu....")
            seespoints = True
            if draw:
                pass
                #self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            else: 
                pass

            consoleLog.PipeLine_Ok("Seen Person with body OwO...")


        else:
            seespoints = False


        return img,seespoints

    