import math


class Server(object):
    def __init__(self, resources, x, y, a, b, history_data):
        self.resources = resources
        self.x = x
        self.y = y
        self.distance_to_center = int(math.sqrt(x * x + y * y))
        self.usage = 0.0
        self.idle_consumption = b
        self.energy_cost_factor = a
        self.usage_list = history_data['USAGE']
        self.heat_list = history_data['HEAT']
        # self.waiting_list = history_data['WAITING_LIST']
        self.is_on = True

    def get_energy_cost(self, t):
        if self.is_on is True:
            return self.usage_list[t] * self.energy_cost_factor + self.idle_consumption
        else:
            return 0.0

    def is_valid_for_job(self, t, job):
        # 完全没必要考虑 waiting list，因为基于历史数据，全部过程是决定性的，我们判断本质不会影响过程
        for i in range(job.process_time):
            if self.usage_list[i + t] * self.resources + job.resources < self.resources:
                continue
            else:
                return False
        return True
