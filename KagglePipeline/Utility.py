class Timing:
    """
    Utility class to time the notebook while running. 
    """
    def __init__(self, start_time):
        self.start_time = start_time
        self.counter = 0

    def timer(self, message=None):
        """
        Timing function that returns the time taken for this step since the starting time. Message is optional otherwise we use a counter. 
        """
        if message:
            print(f"{message} at {time.time()-self.start_time}")
        else:
            print(f"{self.counter} at {time.time()-self.start_time}")
            self.counter += 1
        return

