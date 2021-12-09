"""
this is where the const go for the module
"""
from  configparser import ConfigParser

zmq_recv = "tcp://" + "127.0.0.1:5000"
zmq_sender = "tcp://" + "127.0.0.1:5002"


hostname = "127.0.0.1"  # Use to receive from localhost
port = 5555


landmarks = str("../SecuServeFiles/shape_predictor_68_face_landmarks.dat")
