from rllab.algos.ddpg import DDPG
class Model(object):
    def __init__(self):
        self.model = None
    def predict(self, obs):
        pass
    def train(self, batch_data):
        pass


class DDPGModel(Model):
    def __init__(self):
        self.ddpg = DDPG()

    def predict(self, obs):
        action = self.ddpg.policy.get_action(observation=obs)

    def train(self, batch_data):
        self.ddpg.train(batch_data=)

from rllab.algos.trpo import TRPO

class PpoModel(Model):
    def __init__(self):
        self.model = TRPO()


class Agent(object):
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        self.model.predict(obs)



if __name__ == '__main__':


    agent = Agent(model=DDPGModel())
    agent = Agent(model=PpoModel())
