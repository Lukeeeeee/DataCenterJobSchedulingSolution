# The data at each time should include:
#   server[1..n]: usage, heat
#   job: whether generate a new job at this time
#   data center: overall heat
import math
import random
import json
import datetime

T = 2000
sim_t = 1500
process_t_lower = 1
process_t_upper = 30
job_resource_lower = 5
job_resource_upper = 20
server_r = 40
server_count = 5
current_time = 0
a = 5.0
b = 10.0
server_heat_constant = 100

alpha = 0.5
beta = 0.5

ti = datetime.datetime.now()
log_dir = (
str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute) + '-' + str(ti.second) + '.data')


class Server(object):
    def __init__(self, x, y, no):
        self.id = no
        self.r = server_r
        self.x = x
        self.y = y
        self.dis_to_center = math.sqrt(x * x + y * y)
        self.queue = []
        self.usage_list = [0.0 for _ in range(T)]
        self.energy_list = [0.0 for _ in range(T)]
        self.current_job = []
        self.is_valid = [False for _ in range(T)]
        self.heat_list = [0.0 for _ in range(T)]

    def check_with_job(self, job):
        for i in range(current_time, T):
            flag = True
            for j in range(job.process_t):
                if i + j < T and self.usage_list[i + j] * self.r + job.r <= self.r:
                    continue
                else:
                    flag = False
                    break
            if flag is True:
                return True, i
        return False, -1

    def add_job(self, job):
        res, start_t = self.check_with_job(job)
        if res is False:
            return False
        else:
            self.queue.append((job, start_t))
            job.host_server = self.id
            job.start_t = start_t
            job.finish_t = job.process_t + start_t - 1
            for i in range(start_t, job.process_t + start_t):
                self.usage_list[i] = self.usage_list[i] + job.r / (self.r * 1.0)
                self.energy_list[i] = self.usage_list[i] * a + b * (self.usage_list[i] > 0.000001)
            return True

    def step(self):
        # Every time step, update queue, current_job, do not update usage list.
        for job, st in self.queue:
            if st == current_time:
                self.current_job.append(job)
                self.queue.remove((job, st))

        for job in self.current_job:
            job.left_time = job.left_time - 1
            if job.left_time == 0.0:
                self.current_job.remove(job)
        self.heat_list[current_time] = self.energy_list[current_time] * server_heat_constant * \
                                       (1.0 - self.dis_to_center / 10.0)

    def return_state_dict(self, ti):
        return {
            "X": self.x,
            "Y": self.y,
            "USAGE": self.usage_list[ti],
            "ENERGY": self.energy_list[ti],
            "IS_VALID": int(self.is_valid[ti]),
            "RESOURCE": self.r,
            "HEAT": self.heat_list[ti],
            # 'ID': self.id
        }


class Job(object):
    def __init__(self, sub_t, ddl, process_t, r):
        self.sub_t = sub_t
        self.start_t = -1
        self.ddl = ddl
        self.process_t = process_t
        self.r = r
        self.left_time = self.process_t
        self.finish_t = -1
        self.host_server = -1

    def return_state_dict(self):
        return {
            "SUBMIT_TIME": self.sub_t,
            "DEADLINE": self.ddl,
            "PROCESS_TIME": self.process_t,
            "RESOURCES": self.r,
            "FINISH_TIME": self.finish_t,
            # "HOST_SERVER": self.host_server
        }


class DataCenter(object):
    def __init__(self):
        self.heat_list = [0.0 for _ in range(T)]

    def step(self, server_list):
        total_heat = 0.0
        for server in server_list:
            total_heat = total_heat + server.heat_list[current_time]
        if current_time > 0:
            total_heat = total_heat + self.heat_list[current_time - 1] * 0.6
        self.heat_list[current_time] = total_heat

    def return_state_dict(self, ti):
        return {
            "TOTAL_HEAT": self.heat_list[ti]
        }


class Reward(object):
    def __init__(self):
        self.reward_list = [0.0 for _ in range(T)]

    def compute_reward(self, server_list, dc):

        for i in range(T):
            efficiency_sum = 0.0
            for server in server_list:
                efficiency_sum = efficiency_sum + (server.usage_list[i] + 0.00000001) / (
                server.energy_list[i] + 0.00000001)

            if i > 0:
                delta_heat = dc.heat_list[i - 1] - dc.heat_list[i]
            else:
                delta_heat = -dc.heat_list[i]
            self.reward_list[i] = efficiency_sum * alpha + beta * delta_heat

    def return_reward_dict(self, ti):
        return {
            "REWARD": self.reward_list[ti],
            # "TIME": ti
        }


def get_state(ti, server_list, job_list, dc):
    state = {}
    reward_state_dict = {}
    for se in server_list:
        reward_state_dict[str(se.id)] = se.return_state_dict(ti)
    state['SERVER_STATE'] = reward_state_dict
    state['JOB_STATE'] = job_list[ti].return_state_dict()
    state['DC'] = dc.return_state_dict(ti)
    # state['TIME'] = ti
    return state


def get_one_sample(ti, server_list, job_list, dc, re):
    state = get_state(ti, server_list, job_list, dc)
    state_next = get_state(ti + 1, server_list, job_list, dc)

    reward = re.return_reward_dict(ti)
    return {
        "TIME": ti,
        "STATE": state,
        "ACTION": job_list[ti].host_server,
        "REWARD": reward,
        "NEXT_STATE": state_next
    }


def main():
    global current_time
    job_list = []
    server_list = []
    x = [-1, -1, 0, 1, 1]
    y = [-1, 1, 0, -1, 1]
    dc = DataCenter()
    for i in range(5):
        server = Server(x=x[i], y=y[i], no=i)
        server_list.append(server)

    for i in range(sim_t):
        print("Time at %d\n" % i)
        sub_t = current_time
        process_t = int(random.uniform(process_t_lower, process_t_upper))
        ddl = int(random.uniform(current_time + process_t_upper, T))
        r = int(random.uniform(job_resource_lower, job_resource_upper))
        new_job = Job(sub_t=sub_t, ddl=ddl, process_t=process_t, r=r)
        job_list.append(new_job)
        valid_server = []
        for server in server_list:
            res, _ = server.check_with_job(new_job)
            if res is True:
                server.is_valid[current_time] = True
                valid_server.append(server.id)
            print(server.is_valid[current_time])
        if len(valid_server) == 0:
            raise Exception("Error resoruces")
            pass
        else:
            random.shuffle(valid_server)
            k = valid_server[0]
            print(k)
            if server_list[k].add_job(new_job) is False:
                raise Exception("Error")
        for server in server_list:
            server.step()
        dc.step(server_list=server_list)
        current_time = current_time + 1
    reward = Reward()
    reward.compute_reward(server_list=server_list, dc=dc)

    sample_set = []

    for i in range(sim_t - 1):
        sample_set.append(get_one_sample(i, server_list, job_list, dc, reward))
    with open(log_dir, 'w') as f:
        json.dump(sample_set, fp=f, indent=4)

if __name__ == '__main__':
    main()
