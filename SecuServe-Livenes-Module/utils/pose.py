""" 
this is a pose detector in tensorflow so i can detect blinking and a skeletion tracking 


"""

import cv2
import mediapipe as mp
import time

class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=False, detectionCon = True, trackCon = 0.3):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        #print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            seespoints = True
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            else: 
                pass

        else:
            seespoints = False


        return img,seespoints

    def getPosition(self, img, draw=True):
        lmList= []

        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                
                if(id >= 11 and id <= 16):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                    print("id"+ str(id)+ " ")
                    print("pos"+ " "+str(cx) +" "+ str(cy))
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        
        return lmList