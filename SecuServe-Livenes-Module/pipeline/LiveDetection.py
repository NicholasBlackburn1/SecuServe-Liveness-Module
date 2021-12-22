"""
this class is for detecting and findign and handling computer vision /tensorflow of detecting faces lively 
#TODO TRY TO GET TENSORFLOW TO RECONIZE BODY POS to see if the user is alive too
"""


import datetime
from os import sendfile
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

    EYE_COUNT = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]




    #* sends status updates from liveness detection
    def sendLifeStatus(self,sender,Alive:bool, hasbody:bool,eyeamount:int):
        sender.send_string("LIVENESS")
        sender.send_json({'alive':Alive,'time':str(datetime.now), 'hasBody':hasbody, 'eyeAmmount':eyeamount})



    # this will send status messages accross the pipeline
    def sendProgramStatus(self, sender,status):
        
        sender.send_string("LIVENESS_STATS")
        sender.send_json({'status': '"'+str(status)+'"','alive':False,'time':str(datetime.now)})


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
                
                self.eyePosDetection(image,ret=rects)
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

                    consoleLog.Debug("sending message to opencv")
                    self.sendLifeStatus(sender=sender, Alive=False, hasbody=False,eyeamount=self.EYE_COUNT)
                    consoleLog.PipeLine_Ok("Set data to opencv")

                else:
                        
                        self.sendLifeStatus(sender=sender, Alive=True, hasbody=False, eyeamount=0)
                    # reset the eye frame counter
                self.COUNTER = 0


    def eyeThresholding(self,value ):  # function to threshold and give either left or right
        
        if (value<=54):   #check the parameter is less than equal or greater than range to 
            left_counter=self.left_counter+1		#increment left counter 

            if (left_counter>self.th_value):  #if left counter is greater than threshold value 
                consoleLog.info('RIGHT EYE')

            if(self.right_counter>self.th_value):
                consoleLog.info('LEFT EYE')
                self.right_counter=0

    #* eye detection and pos location
    def eyePosDetection(self,frame,ret):
        
    
        if ret:
         
            eyesUwU = frame
            pupilFrame=frame
            clahe=frame
           
            eyes = cv2.CascadeClassifier("../SecuServeFiles/haarcascade_eye.xml")
            detected = eyes.detectMultiScale(eyesUwU, 1.36, 5)
           
      
            for (x,y,w,h) in detected: #similar to face detection but for eyes
                

                cv2.rectangle(eyesUwU, (x,y), ((x+w),(y+h)), (0,0,255),1)	 #draw rectangle around eyes
                cv2.line(eyesUwU, (x,y), (x+w,y+h), (0,0,255),1)   #draw cross
                cv2.line(eyesUwU, (x+w,y), (x,y+h), (0,0,255),1)

                pupilFrame = cv2.cvtColor(eyesUwU, cv2.COLOR_BGR2GRAY)

                cl1 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)) #set grid size
                clahe = cl1.apply(pupilFrame)  #clahe
                blur = cv2.medianBlur(clahe, 7)  #median blur

                
                circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=7,maxRadius=21) #houghcircles
            
                if circles is not None: #if atleast 1 is detected

                    circles = self.EYE_COUNT

                    circles = np.round(circles[0, :]).astype("int") #change float to integer
                    self.localinfo(eyesUwU,self.EYE_COUNT)
                 
                    for (x,y,r) in circles:
                       
                        cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                        cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                        #self.eyeThresholding(x)

                cv2.imshow("cle",clahe)

    #* draws info on opencv frame output 
    def localinfo(self,frame,info):
        cv2.putText(
        frame,
        "Eye count"+str(info),
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

