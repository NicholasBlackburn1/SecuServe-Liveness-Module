"""
this is the file holds the Pipeline and controls Pipline
watch dog count to frozen 
"""
from enum import Enum


from zmq.sugar.frame import Message
from pipeline import LiveDetection
from utils import consoleLog
from pipeline import state

from datetime import datetime, time


class States(Enum):
    IDLE = 0
    SETUP_PIPELINE = 1
    TRAIN_MODEL = 2
    RUN_RECONITION = 4
    ERROR = 5


# Start of our state.States
class SetupPipeLine(state.State):
    """
    The state.State which Sets Up Whole opencv pipeline
    """

    def on_event(self, event, sender,tf):
        if event == States.SETUP_PIPELINE:
          
            #TODO: GET FACE LIVE PIPELINE SETUP
            LiveDetection.LiveDetection.pipelineSetUp(LiveDetection,sender=sender,tf=tf)
            # consoleLog.PipeLine_Data("Model last trained"+" "+ str(moddate['%H']))
            self.next_state(States.TRAIN_MODEL)

            return TrainPipeline()

        return self


class TrainPipeline(state.State):
    """
    The state.State which Trains the Reconized face Models
    """

    def on_event(self, event, sender):
        if event == States.TRAIN_MODEL:

            #TODO: Set up training UwU
            self.next_state(States.RUN_RECONITION)
            return RunReconitionPipeLine()

        return self


class RunReconitionPipeLine(state.State):
    """
    The state.State which Reconizes Faces
    """

    def on_event(self, event, sender):
        if event == States.RUN_RECONITION:
            LiveDetection.LiveDetection.runPipeline(LiveDetection,sender=sender)
            
         

        return self


class Idle(state.State):
    """
    The state.State which The program waits for a face to be spotted
    """

    def on_event(self, event, sender,receiver,poller):
        if event == States.IDLE:
            consoleLog.Warning("Idleing....")
            time.sleep(0.5)

        return self


class Error(state.State):
    """
    The state.State which The program waits for a face to be spotted
    """

    msg = None

    def __init__(self, message):
        self.msg = message

    def on_event(self, event, sender):
        if event == States.ERROR:
            videoRequired.consoleLog.Error("ERROR....")
            sender.send_string("ERROR")
            sender.send_json({"error": str(self.msg), "time": str(datetime.now())})
            return

        return self


class PipeLine(object):
    """
    A simple state.State machine that mimics the functionality of a device from a
    high level.
    """

    def __init__(self):
        """Initialize the components."""

        # Start with a default state.State.
        self.State = SetupPipeLine()

    def on_event(self, event, sender,tf,img_receiver):
        """
        This is the bread and butter of the state.State machine. Incoming events are
        delegated to the given state.States which then handle the event. The result is
        then assigned as the new state.State.
        """

        # The next state.State will be the result of the on_event function.
        self.State = self.State.on_event(event, sender,tf,img_receiver)

    def getCurrentStat(self):
        return self.State
