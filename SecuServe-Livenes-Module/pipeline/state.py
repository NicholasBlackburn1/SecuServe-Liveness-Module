"""
State Machine Base class 
"""
from utils import consoleLog


class State(object):
    """
    We define a state object which provides some utility functions for the
    individual states within the state machine.
    """

    def __init__(self):
        consoleLog.PipeLine_Ok("Processing current state:" + str(self))

    def next_state(self, state):
        consoleLog.Warning("Next state:" + str(state))

    def on_event(self, event):
        """
        Handle events that are delegated to this State.
        """
        pass

    def __repr__(self):
        """
        Leverages the __str__ method to describe the State.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns the name of the State.
        """
        return self.__class__.__name__
