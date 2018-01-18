class Environment(object):
    def __init__(self, server_list, job_list, data_center, history_list):
        self.server_list = server_list
        self.job_list = job_list
        self.data_center = data_center
        self.reward_list = history_list['REWARD']

    def step(self, action):
        pass

    def return_mini_batch(self):
        pass
