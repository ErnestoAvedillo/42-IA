import numpy as np
class Event_Type():
    def __init__(self, filename):
        self.event_type = None
        self.filenames = [
            ["R03.edf","R04.edf","R07.edf","R08.edf","R11.edf","R12.edf"],
            ["R05.edf","R06.edf","R09.edf","R10.edf","R13.edf","R14.edf"]
            ]
        self.event_labels = [1, 2, 3]
        self.events = [{"T0": 0, "T1": 1, "T2": 2},{"T0": 0, "T1": 3, "T2": 4}]
        self.set_event_type(filename)

    def set_event_type(self, filename): 
        if any(word in filename for word in self.filenames[0]):
            self.event_type = self.events[0]
        elif any(word in filename for word in self.filenames[1]):
            self.event_type = self.events[1]
        else:
            self.event_type = None
        return self.event_type
    
    def get_event_nr(self, event_label):
        key_events = self.event_type.keys()
        return self.event_type[event_label]
    
    def convert_event_labels(self, Y):
        for key in self.event_labels:
            Y[Y == key] = self.event_type[key]
        return Y