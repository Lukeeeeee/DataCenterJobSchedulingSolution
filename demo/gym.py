from gym.core import Env
from gym.spaces.discrete import Discrete

class DataCenterEnv(Env):
    def __init__(self):
        # load pkl model
        # define the state, action space
        # define reward function
        self.state = None
        self.action_space = Discrete(n=100)
        self.observation_space = None
        self.step_count = 0
        pass

    def _step(self, action):
        # recieve one action
        # self.step_count += 1
        # new_state = self.model.run()
        # reward = self.reward(self,state, next_state, action)
        # done = bool(self.step_count > self.max_step)
        # if done is True:
        #    self.reset()
        # self.state = new_state
        pass


    def _reward(self, state, next_state, action):
        return 0

    def _reset(self):
        pass

    def _seed(self, seed=None):
        pass


class Agent(object):
    def __init__(self):
        self.model = DQN(batch=100, lr=.001)

    def predict(self, obs):
        action = self.model.predict(obs)

        return action


if __name__ == '__main__':
    import gym
    a = gym.make('Swimmer-v1')
    a.reset()
    for i in range(1000):
        print(a.step(action=a.action_space.sample()))
