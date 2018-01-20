import numpy as np
import tensorflow as tf
import tensorlayer as tl
import datetime
from log import LOG_PATH
import os
import src.visualization as vis
from src.config import Config as con
import tensorflow.contrib as tfcontrib

server_count = con.server_count
server_state_dim = con.server_state_dim
total_server_state_dim = con.total_server_state_dim
server_feature_dim = con.server_feature_dim
job_state_dim = con.job_state_dim
dc_state_dim = con.dc_state_dim
action_dim = con.action_dim

# NET SIZE
server_feature_layer1_size = con.server_feature_layer1_size
q_net_layer1_size = con.q_net_layer1_size
q_net_layer2_size = con.q_net_layer2_size

# TRAIN PARAMETERS
gamma = con.gamma
learning_rate = con.learning_rate
batch_size = con.batch_size
epsilon = con.epsilon
update_target_q_every_iter = con.update_target_q_every_iter

ti = datetime.datetime.now()
log_dir = (LOG_PATH + '/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-' + str(ti.minute) + '-' + str(
    ti.second) + '/')
if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        tf.summary.scalar('value', var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Agent(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
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

        self.target_q_net = self.create_q_network(prefix='TARGET_')
        self.target_q = self.target_q_net.outputs

        self.update_target_q_op = self.create_target_update_op_list()

        # Define greedy policy to choose a valid action
        temp = tf.multiply(x=self.action_is_valid,
                           y=tf.constant(1000.0, shape=[batch_size, server_count]))
        self.temp = tf.add(x=self.q, y=temp)

        self.greedy_policy_action = tf.argmax(self.temp, axis=1)

        # Define op for q and target q with corresponding action
        self.q_off_by_action = tf.multiply(self.q, tf.cast(self.action_one_hot, tf.float32))
        # self.q_off_by_action = self.q

        self.target_q_off_by_action = tf.multiply(self.reward_input + gamma * self.q,
                                                  tf.cast(self.action_one_hot, tf.float32))
        # self.target_q_off_by_action = self.reward_input + gamma * self.target_q,

        self.loss, self.optimizer, self.optimize_op, self.compute_gradients_op = self.create_training_method(
            target_q_off_by_action=self.target_q_off_by_action_input)

        self.gradients = self.optimizer.compute_gradients(loss=self.loss)

        # Some op for test and visualization

        self.max_q = tf.reduce_max(self.q, axis=1)
        self.action = tf.argmax(self.q, axis=1)

        self.mean_max_q = tf.reduce_mean(self.max_q)
        variable_summaries(self.mean_max_q, 'mean_q')

        # variable_summaries(self.compute_gradients_op, 'gradients')

        # variable_summaries(self.loss, 'loss')
        self.merged_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # Init op

        tl.layers.initialize_global_variables(sess=self.sess)
        self.q_net.print_params()
        self.q_net.print_layers()

    # def eplison_greedy_action_selection(self):
    #     temp = tf.multiply(x=self.action_is_valid,
    #                        y=tf.constant(1000.0, shape=[batch_size, server_count]))
    #     self.temp = tf.add(x=self.q, y=temp)
    #     unpacked_q = tf.unstack(self.temp, axis=0)
    #
    #     greedy_policy_action_list = []
    #
    #     for tensor in unpacked_q:
    #         if np.random.uniform(0, 1.0) < epsilon:
    #             greedy_policy_action_list.append(tf.argmax(tensor, axis=1))
    #         else:
    #             k = np.random.randint(0, server_count)
    #             greedy_policy_action_list.append(k)
    #     self.greedy_policy_action = tf.argmax(self.temp, axis=1)

    def define_server_feature_extraction_net(self, input, reuse=False, prefix=''):

        with tf.variable_scope("SEVER_STATE", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            server_feature_extraction_net = tl.layers.InputLayer(inputs=input,
                                                                 name=prefix + 'SERVER_STATE_INPUT')
            server_feature_extraction_net = tl.layers.DenseLayer(layer=server_feature_extraction_net,
                                                                 n_units=server_feature_layer1_size,
                                                                 act=tf.nn.leaky_relu,
                                                                 name=prefix + 'SERVER_STATE_LAYER_1')
            server_feature_extraction_net = tl.layers.DenseLayer(layer=server_feature_extraction_net,
                                                                 n_units=server_feature_dim,
                                                                 name=prefix + 'SERVER_STATE_LAYER_2')
            return server_feature_extraction_net

    def create_q_network(self, prefix=''):
        server_state_tensor_list = tf.split(self.server_state_input, server_count, axis=1)
        server_feature_tensor_layer_list = []
        for i in range(server_count):
            tensor = tf.reshape(server_state_tensor_list[i], shape=(-1, server_state_dim))
            if i == 0:
                reuse = False
            else:
                reuse = True

            server_feature_tensor_layer_list.append(self.define_server_feature_extraction_net(input=tensor,
                                                                                              reuse=reuse,
                                                                                              prefix=prefix))
        job_input_layer = tl.layers.InputLayer(inputs=self.job_state_input,
                                               name=prefix + 'JOB_STATE_INPUT')
        dc_input_layer = tl.layers.InputLayer(inputs=self.dc_state_input,
                                              name=prefix + 'DC_STATE_INPUT')

        all_state_layer = tl.layers.ConcatLayer(
            layer=server_feature_tensor_layer_list + [job_input_layer, dc_input_layer],
            concat_dim=1,
            name=prefix + 'SERVER_FEATURE')

        q_net = tl.layers.DenseLayer(layer=all_state_layer,
                                     n_units=q_net_layer1_size,
                                     act=tf.nn.leaky_relu,
                                     name=prefix + 'Q_NET_LAYER_1')
        q_net = tl.layers.DenseLayer(layer=q_net,
                                     n_units=q_net_layer2_size,
                                     act=tf.nn.leaky_relu,
                                     name=prefix + 'Q_NET_LAYER_2')
        q_net = tl.layers.DenseLayer(layer=q_net,
                                     n_units=server_count,
                                     name=prefix + 'Q_NET_LAYER_3')
        return q_net

    def create_training_method(self, target_q_off_by_action):
        loss = tf.reduce_mean(tf.squared_difference(target_q_off_by_action, self.q_off_by_action))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.3)
        optimize = optimizer.minimize(loss=loss, var_list=self.q_net.all_params)
        compute_gradients = optimizer.compute_gradients(loss=loss, var_list=self.q_net.all_params)
        regularizer = tfcontrib.layers.l1_l2_regularizer()
        loss = loss + tfcontrib.layers.apply_regularization(regularizer, weights_list=self.q_net.all_params)
        return loss, optimizer, optimize, compute_gradients

    def create_target_update_op_list(self):
        op = []
        for (q_para, target_q_para) in zip(self.q_net.all_params, self.target_q_net.all_params):
            op.append(target_q_para.assign(q_para))
        return op

    def eval_some_tensor(self, tensor, mini_batch):
        # For test and visual
        res = self.sess.run(fetches=[tensor],
                            feed_dict={
                                self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                self.dc_state_input: mini_batch['STATE']['DC'],
                                self.action_input: mini_batch['ACTION'],
                            })
        return res

    def eval_q_off_by_action(self, state_dict, action):
        return self.sess.run(fetches=[self.q_off_by_action],
                             feed_dict={
                                 self.server_state_input: state_dict['SERVER_STATE'],
                                 self.job_state_input: state_dict['JOB_STATE'],
                                 self.dc_state_input: state_dict['DC'],
                                 self.action_input: action
                             })

    def eval_greedy_policy_action(self, state_dict):
        res, temp = self.sess.run(fetches=[self.greedy_policy_action, self.temp],
                                  feed_dict={
                                      self.server_state_input: state_dict['SERVER_STATE'],
                                      self.job_state_input: state_dict['JOB_STATE'],
                                      self.dc_state_input: state_dict['DC'],
                                      self.action_is_valid: state_dict['VALID_ACTION']
                                  })
        return np.reshape(np.array(res), [-1])

    def eval_action(self, state_dict):
        # For test and visual
        res = self.sess.run(fetches=[self.action],
                            feed_dict={
                                self.server_state_input: state_dict['SERVER_STATE'],
                                self.job_state_input: state_dict['JOB_STATE'],
                                self.dc_state_input: state_dict['DC'],
                                self.action_is_valid: state_dict['VALID_ACTION']
                            })
        return np.reshape(np.array(res), [-1])

    def eval_target_q_off_by_action(self, next_state_dict, next_action, reward):
        res = self.sess.run(fetches=[self.target_q_off_by_action],
                            feed_dict={
                                self.reward_input: reward,
                                self.server_state_input: next_state_dict['SERVER_STATE'],
                                self.job_state_input: next_state_dict['JOB_STATE'],
                                self.dc_state_input: next_state_dict['DC'],
                                self.action_input: next_action
                            })
        return np.reshape(np.array(res), newshape=[-1, server_count])

    def eval_gradients(self, mini_batch):
        next_action = self.eval_greedy_policy_action(state_dict=mini_batch['NEXT_STATE'])

        target_q_off_by_action = self.eval_target_q_off_by_action(next_state_dict=mini_batch['NEXT_STATE'],
                                                                  next_action=next_action,
                                                                  reward=mini_batch['REWARD'])
        gradients = self.sess.run(fetches=[self.compute_gradients_op],
                                  feed_dict={
                                      self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                      self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                      self.dc_state_input: mini_batch['STATE']['DC'],
                                      self.action_input: mini_batch['ACTION'],
                                      self.target_q_off_by_action_input: target_q_off_by_action
                                  })
        return gradients

    def train(self, mini_batch):

        next_action = self.eval_greedy_policy_action(state_dict=mini_batch['NEXT_STATE'])

        target_q_off_by_action = self.eval_target_q_off_by_action(next_state_dict=mini_batch['NEXT_STATE'],
                                                                  next_action=next_action,
                                                                  reward=mini_batch['REWARD'])

        _, loss = self.sess.run(fetches=[self.optimize_op, self.loss],
                                feed_dict={
                                    self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                    self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                    self.dc_state_input: mini_batch['STATE']['DC'],
                                    self.action_input: mini_batch['ACTION'],
                                    self.target_q_off_by_action_input: target_q_off_by_action
                                })
        # gradients = self.sess.run(fetches=[self.compute_gradients_op],
        #                           feed_dict={
        #                               self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
        #                               self.job_state_input: mini_batch['STATE']['JOB_STATE'],
        #                               self.dc_state_input: mini_batch['STATE']['DC'],
        #                               self.action_input: mini_batch['ACTION'],
        #                               self.target_q_off_by_action_input: target_q_off_by_action
        #                           })
        # print(target_q_off_by_action)
        # print(self.eval_some_tensor(tensor=self.q_off_by_action, mini_batch=mini_batch))
        # print(self.eval_some_tensor(tensor=self.reward_input, mini_batch=mini_batch))
        # print(self.eval_some_tensor(tensor=self.target_q_off_by_action))
        # print (gradients)

        return loss

    def update_target_net(self):

        res = self.sess.run(self.update_target_q_op)
        # res = self.sess.run(self.target_q_net.all_params[0])
        # print(res)

    def do_summary(self, mini_batch, epoch):
        summary = self.sess.run(fetches=[self.merged_summary, self.max_q, self.action],
                                feed_dict={
                                    self.server_state_input: mini_batch['STATE']['SERVER_STATE'],
                                    self.job_state_input: mini_batch['STATE']['JOB_STATE'],
                                    self.dc_state_input: mini_batch['STATE']['DC'],
                                    self.action_input: mini_batch['ACTION']
                                })
        self.file_writer.add_summary(summary=summary[0], global_step=epoch)


training_data_list = []


def do_print(test_batch, epoch, iter, print_flag=False):
    global training_data_dict
    server_state = np.array(test_batch['STATE']['SERVER_STATE'])
    action = a.eval_action(state_dict=test_batch['STATE'])
    q = a.eval_some_tensor(a.q, mini_batch=test_batch)[0]
    q_off_by_action = a.eval_some_tensor(tensor=a.q_off_by_action, mini_batch=test_batch)
    next_action = a.eval_greedy_policy_action(state_dict=test_batch['NEXT_STATE'])

    target_q_off_by_action = a.eval_target_q_off_by_action(next_state_dict=test_batch['NEXT_STATE'],
                                                           next_action=next_action,
                                                           reward=test_batch['REWARD'])
    grad = a.eval_gradients(test_batch)

    if print_flag is True:
        print("choosed action", action)
        print("Q", q)
        print("Input Action", test_batch['ACTION'])
        print("Q off by action", q_off_by_action)
        print ("target Q off by action", target_q_off_by_action)
    dict = {
        'EPOCH': epoch,
        'ITER': iter,
        'SERVER_STATE': server_state,
        'ACTION': action,
        'Q': q
    }
    training_data_list.append(dict)
    pass

if __name__ == '__main__':

    from src.environment import Environment

    global training_data_list
    import src.visualization as vis

    a = Agent()

    env = Environment(file_name="1-21-1-21-57.data")

    batch_iter = con.batch_iter
    epoch = con.epoch

    for T in range(epoch):
        print("Epoch %d" % T)
        total_loss = 0.0
        for i in range(batch_iter):
            if i % update_target_q_every_iter == 0:
                a.update_target_net()
            data_batch = env.return_mini_batch(i, batch_size)
            loss = a.train(mini_batch=data_batch)
            total_loss = total_loss + loss
            if T % con.save_data_every_epoch == 0:
                do_print(test_batch=data_batch, epoch=T, iter=i, print_flag=True)
        print("Aver loss = %f" % (total_loss / batch_iter))
    res = np.array(training_data_list)
    np.save(file=log_dir + '/training_data', arr=res)
    vis.visual(res)
