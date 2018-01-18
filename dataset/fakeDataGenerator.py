# The data at each time should include:
#   server[1..n]: usage, heat
#   job: whether generate a new job at this time
#   data center: overall heat
import math
import random

T = 200
process_t_lower = 1
process_t_upper = 20
job_resource_lower = 4
job_resource_upper = 10
server_r = 20
server_count = 5
current_time = 0


class Server(object):
    def __init__(self, x, y):
        self.r = server_r
        self.x = x
        self.y = y
        self.queue = []
        self.usage_list = [0.0 for _ in range(T)]
        self.current_job = []

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
            job.start_t = start_t
            for i in range(start_t, job.process_t + start_t):
                self.usage_list[i] = self.usage_list[i] + job.r / (self.r * 1.0)
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
                job.finish_time = current_time
                self.current_job.remove(job)


class Job(object):
    def __init__(self, sub_t, ddl, process_t, r):
        self.sub_t = sub_t
        self.start_t = -1
        self.ddl = ddl
        self.process_t = process_t
        self.r = r
        self.left_time = self.process_t
        self.finish_t = -1


def main():
    global current_time
    job_list = []
    server_list = []
    x = [-1, -1, 0, 1, 1]
    y = [-1, 1, 0, -1, 1]
    for i in range(5):
        server = Server(x=x[i], y=y[i])
        server_list.append(server)

    for i in range(T):
        print("Time at %d\n" % i)
        if i % 10 == 0:
            sub_t = current_time
            process_t = int(random.uniform(process_t_lower, process_t_upper))
            ddl = int(random.uniform(current_time + process_t_upper, T))
            r = int(random.uniform(job_resource_lower, job_resource_upper))
            new_job = Job(sub_t=sub_t, ddl=ddl, process_t=process_t, r=r)
            job_list.append(new_job)
            flag = False
            for server in server_list:
                if server.add_job(new_job):
                    flag = True
                    break
            if flag is False:
                raise Exception("Error resoruces")
        for server in server_list:
            server.step()
        current_time = current_time + 1


if __name__ == '__main__':
    main()
