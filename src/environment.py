from dataset import DATASET_PATH
import random
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from src.config import Config as con

server_count = con.server_count
server_state_dim = con.server_state_dim
# [R, ENERGY, Y, HEAT, IS_VALID, USAGE, X]
total_server_state_dim = con.total_server_state_dim
server_feature_dim = con.server_feature_dim
job_state_dim = con.job_state_dim
dc_state_dim = con.dc_state_dim
action_dim = con.action_dim


def check_data(sample):
    # Check heat
    total_heat = 0.0
    for i in range(server_count):
        total_heat = total_heat + sample['NEXT_STATE']['SERVER_STATE'][str(i)]['HEAT']
    total_heat = total_heat + 0.6 * sample['STATE']['DC']['TOTAL_HEAT']
    if math.fabs(total_heat - sample['NEXT_STATE']['DC']['TOTAL_HEAT']) > 0.01:
        raise Exception('heat computing error %d' % math.fabs(total_heat - sample['NEXT_STATE']['DC']['TOTAL_HEAT']))
    pass


class Environment(object):
    def __init__(self, file_name):
        with open(DATASET_PATH + '/' + file_name, mode='r') as f:
            self.dataset = json.load(fp=f)

        self.handled_dataset = []
        random.shuffle(self.dataset)
        for sample in self.dataset:
            check_data(sample)
            new_sample = dict()
            new_sample['STATE'] = self.handle_state_dict(state_dict=sample['STATE'])
            new_sample['NEXT_STATE'] = self.handle_state_dict(state_dict=sample['NEXT_STATE'])
            new_sample['REWARD'] = sample['REWARD']['REWARD']
            new_sample['ACTION'] = sample['ACTION']
            self.handled_dataset.append(new_sample)

        self.handled_dataset = self.standardization()
        self.sample_count = len(self.dataset)

    def return_mini_batch(self, index, batch_size):
        batch_data = self.handled_dataset[index * batch_size: (index + 1) * batch_size]
        batch_dict = {
            'STATE': {
                'JOB_STATE': [],
                'SERVER_STATE': [],
                'DC': []
            },
            'NEXT_STATE': {
                'JOB_STATE': [],
                'SERVER_STATE': [],
                'DC': []
            },
            'ACTION': [],
            'REWARD': []
        }
        pass
        batch_dict['STATE']['JOB_STATE'] = [batch_data[i]['STATE']['JOB_STATE'] for i in range(batch_size)]
        batch_dict['STATE']['SERVER_STATE'] = [batch_data[i]['STATE']['SERVER_STATE'] for i in range(batch_size)]
        batch_dict['STATE']['DC'] = [batch_data[i]['STATE']['DC'] for i in range(batch_size)]
        batch_dict['STATE']['VALID_ACTION'] = [batch_data[i]['STATE']['VALID_ACTION'] for i in range(batch_size)]

        batch_dict['NEXT_STATE']['JOB_STATE'] = [batch_data[i]['NEXT_STATE']['JOB_STATE'] for i in range(batch_size)]
        batch_dict['NEXT_STATE']['SERVER_STATE'] = [batch_data[i]['NEXT_STATE']['SERVER_STATE'] for i in
                                                    range(batch_size)]
        batch_dict['NEXT_STATE']['DC'] = [batch_data[i]['NEXT_STATE']['DC'] for i in range(batch_size)]
        batch_dict['NEXT_STATE']['VALID_ACTION'] = [batch_data[i]['NEXT_STATE']['VALID_ACTION'] for i in
                                                    range(batch_size)]

        batch_dict['ACTION'] = [batch_data[i]['ACTION'] for i in range(batch_size)]
        batch_dict['REWARD'] = [batch_data[i]['REWARD'] for i in range(batch_size)]

        for i in range(len(batch_dict['REWARD'])):
            batch_dict['REWARD'][i] = [batch_dict['REWARD'][i] for _ in range(server_count)]

        return batch_dict

    def handle_state_dict(self, state_dict):
        final_state = dict()
        final_state['SERVER_STATE'] = np.array([state_dict['SERVER_STATE'].values()[i].values()
                                                for i in range(server_count)])
        final_state['JOB_STATE'] = np.array(state_dict['JOB_STATE'].values())
        final_state['DC'] = np.array(state_dict['DC'].values())
        final_state['VALID_ACTION'] = np.array([state_dict['SERVER_STATE'].values()[i]['IS_VALID']
                                                for i in range(server_count)])

        return final_state

    def standardization(self):
        # SERVER STATE
        scaler = MinMaxScaler()
        temp_list = []
        for sample in self.handled_dataset:
            temp_list.append(sample['STATE']['SERVER_STATE'])
            temp_list.append(sample['NEXT_STATE']['SERVER_STATE'])

        temp_list = np.array(temp_list).reshape([-1, server_state_dim], order='C')
        scaler.fit(temp_list)
        temp_list = scaler.transform(temp_list)
        temp_list = np.reshape(temp_list, newshape=[-1, 2, server_count, server_state_dim])
        for i in range(len(self.handled_dataset)):
            self.handled_dataset[i]['STATE']['SERVER_STATE'] = temp_list[i][0]
            self.handled_dataset[i]['NEXT_STATE']['SERVER_STATE'] = temp_list[i][1]

        # JOB STATE

        temp_list = []
        for sample in self.handled_dataset:
            temp_list.append(sample['STATE']['JOB_STATE'])
            temp_list.append(sample['NEXT_STATE']['JOB_STATE'])

        temp_list = np.array(temp_list).reshape([-1, job_state_dim], order='C')
        scaler.fit(temp_list)
        temp_list = scaler.transform(temp_list)
        temp_list = np.reshape(temp_list, newshape=[-1, job_state_dim])
        for i in range(len(self.handled_dataset)):
            self.handled_dataset[i]['STATE']['JOB_STATE'] = temp_list[i]
            self.handled_dataset[i]['NEXT_STATE']['JOB_STATE'] = temp_list[i]

        # DC STATE
        temp_list = []
        for sample in self.handled_dataset:
            temp_list.append(sample['STATE']['DC'])
            temp_list.append(sample['NEXT_STATE']['DC'])

        temp_list = np.array(temp_list).reshape([-1, 1], order='C')
        scaler.fit(temp_list)
        temp_list = scaler.transform(temp_list)
        temp_list = np.reshape(temp_list, newshape=[-1, 1])
        for i in range(len(self.handled_dataset)):
            self.handled_dataset[i]['STATE']['DC'] = temp_list[i]
            self.handled_dataset[i]['NEXT_STATE']['DC'] = temp_list[i]
        return self.handled_dataset


if __name__ == '__main__':
    file_name = "1-20-2-21-40.data"
    a = Environment(file_name)
    t = a.return_mini_batch(0, 10)
    pass
