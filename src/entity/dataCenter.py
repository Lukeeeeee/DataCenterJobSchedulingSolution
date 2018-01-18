class DataCenter(object):
    def __init__(self, history_data):
        self.heat_list = history_data['HEAT']

    def get_heat(self, t):
        return self.heat_list[t]
