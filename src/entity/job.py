class Job(object):
    def __init__(self, submit_time, deadline, process_time, resources):
        self.submit_time = submit_time
        self.deadline = deadline
        self.process_time = process_time
        self.resources = resources

    def __cmp__(self, other):
        if self.submit_time > other.submit_time:
            return 1
        elif self.submit_time == other.submit_time:
            return 0
        else:
            return -1
        pass
