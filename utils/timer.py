import time

import torch


class Timer:
    def __init__(self, starting_msg=None):
        """
        This is a timer to estimate the time some step use.
        :param starting_msg: str:Deploy a message on the console.
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start = time.time()
        self.stage_start = self.start
        self.elapsed = None
        self.est_total = None
        self.est_remaining = None
        self.est_finish = None

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        """
        Update the progress of the Timer, which includes the estimation of the total time and remaining time.
        :param progress: progress is a float between 0-1, which means the progress of the entire work.
        e.g progress=0.25 means that the program runs about 25%, 75% remains to finish.
        elapsed: current time minus begin time
        'est' means estimate, this function estimate total time and remaining time.
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        # int do an approximates of finish time.
        self.est_finish = int(self.start + self.est_total)

    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        """
        :return: float(time):the time this stage costs.
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time() - self.stage_start

    def reset_stage(self):
        """
        reclock the stage
        """
        self.stage_start = time.time()

    def lapse(self):
        """
        :return: float(time): when begin a new stage, execute this function.
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

    @staticmethod
    def get_current_time():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time()
