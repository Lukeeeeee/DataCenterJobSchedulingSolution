import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib as contrib

server_count = 5
server_state_dim = 7
total_server_state_dim = server_count * server_state_dim
server_feature_dim = 4
job_state_dim = 5
dc_state_dim = 1
action_dim = server_count

# NET SIZE
server_feature_layer1_size = 10
q_net_layer1_size = 40
q_net_layer2_size = 10

# TRAIN PARAMETERS
gamma = 0.8
learning_rate = 0.01
batch_size = 10


class Agent(object):
    def __init__(self):
        self.sess = tf.Session()
        self.server_state_input = tf.placeholder(tf.float32, shape=[None, server_count, server_state_dim])
        # self.server_state_input_flatten = contrib.layers.flatten(inputs=self.server_state_input)
        self.job_state_input = tf.placeholder(tf.float32, shape=[None, job_state_dim])
        self.dc_state_input = tf.placeholder(tf.float32, shape=[None, dc_state_dim])
        self.action_input = tf.placeholder(tf.uint8, shape=[None])
        self.reward_input = tf.placeholder(tf.float32, shape=[None, server_count])
        self.action_is_valid = tf.placeholder(tf.float32, shape=[None, server_count])

        self.target_q_off_by_action_input = tf.placeholder(tf.float32, shape=[None, server_count])

        self.action_one_hot = tf.one_hot(indices=self.action_input, depth=server_count)

        self.q_net = self.create_q_network()
        self.q = self.q_net.outputs

        temp = tf.multiply(x=self.action_is_valid,
                           y=tf.constant(1000.0, shape=[batch_size, server_count]))
        temp = tf.add(x=self.q, y=temp)

        # self.greedy_policy_action = tf.one_hot(indices=tf.argmax(input=temp, axis=1),
        #                                        depth=server_count)
        self.greedy_policy_action = tf.argmax(temp, axis=1)
        self.q_off_by_action = tf.multiply(self.q, tf.cast(self.action_one_hot, tf.float32))

        self.target_q_off_by_action = tf.multiply(self.reward_input + gamma * self.q,
                                                  tf.cast(self.action_one_hot, tf.float32))

        self.loss, self.optimizer, self.optimize_op, self.compute_gradients_op = self.create_training_method(
            target_q_off_by_action=self.target_q_off_by_action_input)

        self.gradients = self.optimizer.compute_gradients(loss=self.loss)

        # Some op for test and visualization

        self.max_q = tf.reduce_max(self.q, axis=1)
        self.mean_max_q = tf.reduce_mean(self.max_q)

        # Init op

        tl.layers.initialize_global_variables(sess=self.sess)

        self.loss_history = []

    def eval_q_off_by_action(self, state_dict, action):
        return self.sess.run(fetches=[self.q_off_by_action],
                             feed_dict={
                                 self.server_state_input: state_dict['SERVER_STATE'],
                                 self.job_state_input: state_dict['JOB_STATE'],
                                 self.dc_state_input: state_dict['DC'],
                                 self.action_input: action
                             })

    def eval_action(self, state_dict):
        return self.sess.run(fetches=[self.greedy_policy_action],
                             feed_dict={
                                 self.server_state_input: state_dict['SERVER_STATE'],
                                 self.job_state_input: state_dict['JOB_STATE'],
                                 self.dc_state_input: state_dict['DC'],
                                 self.action_is_valid: state_dict['VALID_ACTION']
                             })

    def eval_target_q_off_by_action(self, next_state_dict, next_action, reward):
        return self.sess.run(fetches=[self.target_q_off_by_action],
                             feed_dict={
                                 self.reward_input: reward,
                                 self.server_state_input: next_state_dict['SERVER_STATE'],
                                 self.job_state_input: next_state_dict['JOB_STATE'],
                                 self.dc_state_input: next_state_dict['DC'],
                                 self.action_input: next_action
                             })

    def define_server_feature_extraction_net(self, input, reuse=False):

        with tf.variable_scope("SEVER_STATE", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            server_feature_extraction_net = tl.layers.InputLayer(inputs=input,
                                                                 name='SERVER_STATE_INPUT')
            server_feature_extraction_net = tl.layers.DenseLayer(layer=server_feature_extraction_net,
                                                                 n_units=server_feature_layer1_size,
                                                                 act=tf.nn.relu,
                                                                 name='SERVER_STATE_LAYER_1')
            server_feature_extraction_net = tl.layers.DenseLayer(layer=server_feature_extraction_net,
                                                                 n_units=server_feature_dim,
                                                                 act=tf.nn.relu,
                                                                 name='SERVER_STATE_LAYER_2')
            return server_feature_extraction_net

    def create_q_network(self):
        server_state_tensor_list = tf.split(self.server_state_input, server_count, axis=1)
        server_feature_tensor_layer_list = []
        for i in range(server_count):
            tensor = tf.reshape(server_state_tensor_list[i], shape=(-1, server_state_dim))
            if i == 0:
                reuse = False
            else:
                reuse = True

            server_feature_tensor_layer_list.append(self.define_server_feature_extraction_net(input=tensor,
                                                                                              reuse=reuse))
        job_input_layer = tl.layers.InputLayer(inputs=self.job_state_input,
                                               name='JOB_STATE_INPUT')
        dc_input_layer = tl.layers.InputLayer(inputs=self.dc_state_input,
                                              name='DC_STATE_INPUT')

        all_state_layer = tl.layers.ConcatLayer(
            layer=server_feature_tensor_layer_list + [job_input_layer, dc_input_layer],
            concat_dim=1,
            name='SERVER_FEATURE')

        q_net = tl.layers.DenseLayer(layer=all_state_layer,
                                     n_units=q_net_layer1_size,
                                     act=tf.nn.relu,
                                     name='Q_NET_LAYER_1')
        q_net = tl.layers.DenseLayer(layer=q_net,
                                     n_units=q_net_layer2_size,
                                     act=tf.nn.relu,
                                     name='Q_NET_LAYER_2')
        q_net = tl.layers.DenseLayer(layer=q_net,
                                     n_units=server_count,
                                     name='Q_NET_LAYER_3')
        return q_net

    def create_training_method(self, target_q_off_by_action):
        loss = tf.reduce_mean(tf.squared_difference(target_q_off_by_action, self.q_off_by_action))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimize = optimizer.minimize(loss=loss)
        compute_gradients = optimizer.compute_gradients(loss=loss)
        return loss, optimizer, optimize, compute_gradients

    def eval_some_tensor(self, tensor, mini_batch):
        res = self.sess.run(fetches=[tensor],
                            feed_dict={
                                self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                self.dc_state_input: mini_batch['STATE']['DC'],
                                self.action_input: mini_batch['ACTION'],
                            })
        return res

    def train(self, mini_batch):

        next_action = self.eval_action(state_dict=mini_batch['NEXT_STATE'])
        next_action = np.reshape(np.array(next_action), [-1])
        for i in range(len(mini_batch['REWARD'])):
            mini_batch['REWARD'][i] = [mini_batch['REWARD'][i] for _ in range(server_count)]

        target_q_off_by_action = self.eval_target_q_off_by_action(next_state_dict=mini_batch['NEXT_STATE'],
                                                                  next_action=next_action,
                                                                  reward=mini_batch['REWARD'])
        target_q_off_by_action = np.reshape(np.array(target_q_off_by_action), newshape=[-1, server_count])

        _, loss = self.sess.run(fetches=[self.optimize_op, self.loss],
                                feed_dict={
                                    self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                    self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                    self.dc_state_input: mini_batch['STATE']['DC'],
                                    self.action_input: mini_batch['ACTION'],
                                    self.target_q_off_by_action_input: target_q_off_by_action
                                })
        gradients = self.sess.run(fetches=[self.compute_gradients_op],
                                  feed_dict={
                                      self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                      self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                      self.dc_state_input: mini_batch['STATE']['DC'],
                                      self.action_input: mini_batch['ACTION'],
                                      self.target_q_off_by_action_input: target_q_off_by_action
                                  })
        # print(target_q_off_by_action)
        # print(self.eval_some_tensor(tensor=self.q_off_by_action, mini_batch=mini_batch))
        # print(self.eval_some_tensor(tensor=self.reward_input, mini_batch=mini_batch))
        # print(self.eval_some_tensor(tensor=self.target_q_off_by_action))
        # print (gradients)
        return loss


if __name__ == '__main__':
    from src.environment import Environment

    a = Agent()
    env = Environment(file_name="1-19-22-45-3.data")

    batch_iter = env.sample_count / batch_size
    epoch = 100
    for T in range(epoch):
        # print("Epoch %d" % T)
        total_loss = 0.0
        for i in range(batch_iter - 1):
            loss = a.train(mini_batch=env.return_mini_batch(i, batch_size))
            total_loss = total_loss + loss
            # print("Epoch = %3d, Batch = %3d Loss = %f" % (T, i, loss))
        test_batch = env.return_mini_batch(batch_iter - 1, 10)
        print(a.eval_some_tensor(tensor=a.mean_max_q, mini_batch=test_batch))

        print("Aver loss = %f" % (total_loss / 9))
