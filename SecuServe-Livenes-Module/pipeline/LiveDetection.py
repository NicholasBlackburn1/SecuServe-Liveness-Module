"""
this class is for detecting and findign and handling computer vision /tensorflow of detecting faces lively 
#TODO TRY TO GET TENSORFLOW TO RECONIZE BODY POS to see if the user is alive too
"""


import datetime
from os import sendfile
from random import random
import time
import cv2
import numpy as np
import sys 
import traceback
from utils import consoleLog
import dlib
from utils import const
from scipy.spatial import distance as dist
from imutils import face_utils
from datetime import datetime
from cv2.data import haarcascades

from utils import pose 

class LiveDetection(object):


    COUNTER = 0
    EYE_AR_THRESH = 0.23
    EYE_AR_CONSEC_FRAMES = 2.0
    TOTAL = 0

    left_counter = 0 
    right_counter = 0

    blob_detector = None

    EYE_COUNT = 0.0

    blinked = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]




    #* sends status updates from liveness detection
    def sendLifeStatus(self,sender,Alive:bool, hasbody:bool, timesblinked: int):
        sender.send_string("LIVENESS")
        sender.send_json({'alive':Alive,'time':str(datetime.now()), 'hasBody':hasbody, 'times_blinked':timesblinked})
        time.sleep(.5)



    # this will send status messages accross the pipeline
    def sendProgramStatus(self, sender,status):
        
        sender.send_string("LIVENESS_STATS")
        sender.send_json({'status': '"'+str(status)+'"','alive':False,'time': str(datetime.now())})


    #* sets up pipeline 
    def pipelineSetUp(self,tf,sender):
        
        self.sendProgramStatus(sender, status="Starting to setup Pipeline...")
        
        self.sendProgramStatus(sender, status="Setup Pipeline wass sucessfull!")
        

    
    #* this is where the tensorflow will run our liveness detection 
    def runPipeline(self,img_receiver,sender):

        self.sendProgramStatus(sender,"Starting LifeNess detection")
        consoleLog.info("Starting To run the Liveness pipeline")

        #* this is for decting and seeing that there is a face
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(const.landmarks)

        try:
            consoleLog.Warning("Running Livenesss pipeline")
            self.sendProgramStatus(sender, "Liveness detection is Running")

            #* this llows me to recive frames from my camera over the network
            while True:
                
                msg, frame = img_receiver.receive()
                image = cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # detect faces in the grayscale frame
                rects = detector(gray, 0)
                
                
                #pos = pose.PoseDetector( mode = False, upBody = False, smooth=True, detectionCon = True, trackCon = 0.5)
                
                self.faceLandmarks(rects=rects,predictor=predictor,face_utils=face_utils,gray=gray,image=image,sender=sender, pose = None)


        
        except Exception as ex:
            consoleLog.Error('Python error with no Exception handler:')
            consoleLog.Error('Traceback error:'+str(ex))
            consoleLog.Error(traceback.print_exc())

    # gets the aspect of eyes
    def eye_aspect_ratio(self,eye):
        # compute the euclidean distances between the two sets of
	    # vertical eye landmarks (x, y)-coordinates and the horizontal
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4]) 
        C = dist.euclidean(eye[0], eye[3])
    
    # compute the eye aspect ratio
        return (A + B) / (2.0 * C)

    
    # * this allows me to set the face landmarks on the usrs faces
    def faceLandmarks(self,rects,predictor,face_utils,gray,image, sender, pose):
        for rect in rects:
            #self.eyePosDetection(image,ret=rect)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            
            cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
        
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                    self.blinked +=1
                    
                    consoleLog.Debug("sending message to opencv that eyes blinked")

                    self.sendLifeStatus(sender=sender, Alive=False, hasbody=False, timesblinked=self.blinked)
                    consoleLog.PipeLine_Ok("Set data to opencv")

                else:
                        
                    self.sendLifeStatus(sender=sender, Alive=True, hasbody=False,timesblinked=0)
                # reset the eye frame counter
                    self.COUNTER = 0

                self.COUNTER = 0


                


    def eyeThresholding(self,value ):  # function to threshold and give either left or right
        
        if (value<=54):   #check the parameter is less than equal or greater than range to 
            left_counter=self.left_counter+1		#increment left counter 

            if (left_counter>self.th_value):  #if left counter is greater than threshold value 
                consoleLog.info('RIGHT EYE')

            if(self.right_counter>self.th_value):
                consoleLog.info('LEFT EYE')
                self.right_counter=0

                      

    #* draws info on opencv frame output 
    def localinfo(self,frame,info,title):
        cv2.putText(
        frame,
        str(title)+str(info),
        (200, 200),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    #detects a user skeletion 
    def detectSkeliton(self,pose,frame):

        img, sees = pose.findPose(frame)
        
        cv2.imshow("Image", img)
        return sees

