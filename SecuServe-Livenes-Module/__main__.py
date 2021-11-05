"""
this is the main class for the liveeness dector 
"""

from utils import consoleLog
import zmq

def main():
    consoleLog.Warning("Starting Zmq....")

    context = zmq.Context()
    sender = context.socket(zmq.PUB)
    sender.bind("tcp://" + "127.0.0.1:5000")
    
    consoleLog.PipeLine_Ok("Started Zmq...")
    







if __name__ == '__main__':
    main()