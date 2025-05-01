import numpy as np
class Event_Type():
    def __init__(self, filename):
        """ Definition of possible cases of events:
        0: Rest
        1: Open and close Left fist
        2: Open and close Right fist
        3: Imagine Open and close Left fist
        4: Imagine Open and close Right fist
        5: Open and close both fists
        6: Open and close both feets
        7: Imagine Open and close both fists
        8: Imagine Open and close both feets
        Args:
            filename (str): The filename to check.
        Raises:
            ValueError: If the filename does not match any event type.
        Returns:
            None
        """
        self.event_type = None
        self.filenames = [
            ["R01.edf","R02.edf"],              #Rest
            ["R03.edf","R07.edf","R11.edf"],    #open ad close right or left fsit
            ["R04.edf","R08.edf","R12.edf"],    #imagine open ad close right or left fist
            ["R05.edf","R09.edf","R13.edf"],    #open ad close both fists or feet
            ["R06.edf","R10.edf","R14.edf"]     #imagine open ad close both fists or feet
            ]
        self.event_labels = [1, 2, 3]
        self.events = [{"T0": 1},                       #Rest
                       {"T0": 1, "T1": 2, "T2": 3},     #open ad close right (1) or left fsit (2)
                       {"T0": 1, "T3": 4, "T4": 5},     #imagine open ad close right (3) or left (4) fist 
                       {"T0": 1, "T5": 6, "T6": 7},     #open ad close both fists (5) or feet (6)
                       {"T0": 1, "T7": 8, "T8": 9}]     #imagine open ad close both fists (8) or feet (8)
        self.set_event_type(filename)

    def set_event_type(self, filename):
        """Set the event type based on the filename.
        Args:
            filename (str): The filename to check.
        Raises:
            ValueError: If the filename does not match any event type.
        Returns:

        """
        self.event_type = None
        for i in range(len(self.filenames)):
            if any(word in filename for word in self.filenames[i]):
                self.event_type = self.events[i]
                break
        if self.event_type is None:
            raise ValueError(f"Filename {filename} not found in event type list.")
        return self.event_type
    
    def get_event_nr(self, event_label):
        """
        Get the event number based on the event label.
        Args:
            event_label (str): The event label to check.
        Raises:
            ValueError: If the event label does not match any event type.
        Returns:
            int: The event number corresponding to the event label.
        """
        return self.event_type[event_label]
    
    def convert_event_labels(self, Y):
        """Convert event labels to numerical values.
        Args:
            Y (np.Array): List of event labels.
        Returns:
            np.Array: List of numerical values corresponding to the event labels.
        """
        event_list = np.unique(Y)
        labels = Y.copy()
        keys = list(self.event_type.keys())
        for key in event_list:
            labels[Y == key] = self.event_type[keys[key - 1]]
        return labels

    def get_inverted_event_labels(self):
        """Get the inverted event labels.
        Args:
            None
        Returns:
            dict: A dictionary with the inverted event labels.
        """
        if self.event_type is None:
            raise ValueError("Event type is not set. Please set the event type first.")
        inverted_event_id = {v: k for k, v in self.event_type.items()}
        return inverted_event_id