# Python imports.
import tensorflow as tf
import numpy as np
import random
import tflearn

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents import DQNAgent, DDPGAgent
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

from options.OptionWrapper import OptionWrapper

class OptionAgent(Agent):
    """
    Components:
    1. DQN to select option
    2. Low level controllers for each option
    3. Spectrum method to Generate options
    """
    NAME = "option-agent"
    
    def __init__(self, sess=None, obs_dim=None, action_dim=None, action_bound=None, num_actions=None, num_options=0, gamma=0.99, epsilon=0.05, tau=0.001, name=NAME):
        # TODO: Implement an interface for discrete action space
        Agent.__init__(self, name=name, actions=[])

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.num_actions = num_actions
        if self.num_actions is None:
            self.continuous_action = True
        else:
            self.continuous_action = False
            
        self.epsilon = epsilon
        self.gamma = gamma
        self.update_freq = 1
        self.batch_size = 64
        self.tau = tau

        self.option_b_size = 16
        self.option_freq = 16

        self.num_options = num_options

        self.curr_instances = 0

        # TODO: How can I abstract the high-level control policy?
        # TODO: How can I implement a low-level control policy using the linearSARSA?
        self.high_control_main = QNetwork(self.sess, obs_dim=self.obs_dim, num_options=self.num_options, learning_rate=0.00001, name=self.name+"_high_main")
        self.high_control_target = QNetwork(self.sess, obs_dim=self.obs_dim, num_options=self.num_options, learning_rate=0.00001, name=self.name+"_high_target")


        self.network_params = tf.trainable_variables(scope=self.name+"_high_main")
        self.target_network_params = tf.trainable_variables(scope=self.name+"_high_main")
        self.update_target_params = \
                                    [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                                          tf.multiply(self.target_network_params[i], 1.0 - self.tau))
                                     for i in range(len(self.target_network_params))]

        
        self.reset()

    def act(self, state, reward, train=True):
        # Train the high-level DQN.
        state_d = state.data.flatten()
        if self.total_steps > 0 and self.total_steps % self.option_freq == 0 and self.option_buffer.size() > self.option_b_size and train:
            s, a, r, s2, t, duration = self.option_buffer.sample_op(self.option_b_size)
            loss = self.train(s, a, r, s2, t, duration, self.option_b_size)
            
            
            # print('high_ctrl loss=', loss)

        if self.total_steps > 0 and self.total_steps % self.update_freq == 0 and self.experience_buffer.size() > self.batch_size and train:
            # TODO: How freqeuently should you update the options?
            self.train_options()

        # Save sampled transition to the replay buffer
        if not (self.prev_state is None) and not (self.prev_action is None) and train:
            self.experience_buffer.add((self.prev_state, self.prev_action, reward, state_d, state.is_terminal()))

                
        # Generate options
        if self.total_steps % self.option_freq == 0 and self.experience_buffer.size() > self.option_b_size and len(self.options) < self.num_options:
            option = self.generate_option()
            self.options.append(option)
            
        # Pick an option
        if self.current_options is None:
            self.current_option = self.pick_option(state_d)
            self.num_op_executed[self.current_option] += 1
            self.prev_op_state, self.prev_option = state_d, self.current_option
        else:
            if self.current_options.is_terminal(state_d) or state.is_terminal():
                # Save sampled transition to the replay buffer
                if not (self.prev_op_state is None) and not (self.prev_option is None) and train:
                    # TODO: Discount factor for the Value
                    self.option_buffer.add((self.prev_op_state, self.prev_option, self.op_cumulative_reward, state_d, state.is_terminal(), self.op_num_steps))

                self.current_option = self.pick_option(state_d)                
                self.num_op_executed[self.current_option] += 1

                self.prev_op_state, self.prev_option = state_d, self.current_option

                self.op_cumulative_reward = 0
                self.op_num_steps = 0
                
            else:
                # Contiue on 
                self.op_cumulative_reward += reward * self.gamma
                self.op_num_steps += 1

        # Retrieve an action
        # print('current_option = ', self.current_option)
        # print('#options = ', len(self.options))
        assert(self.current_option < len(self.options))
        
        prim_action = self.options[self.current_option].act(state)

        self.prev_state, self.prev_action = state_d, prim_action

        self.curr_step += 1
        self.total_steps += 1

        self.total_reward += reward
        
        if state.is_terminal():
            print('#Episode=', self.curr_episodes, '#steps=', self.curr_step, 'Total_reward=', self.total_reward)
            print('#Options executed = ', self.num_op_executed)
            self.curr_step = 0
            self.curr_episodes += 1
            self.total_reward = 0
            self.current_option = None
            self.op_cumulative_reward = 0
            self.op_num_steps = 0
            
        
        return prim_action

    def pick_option(self, state_d):
        applicable_options = self.get_applicable_options(state_d)

        # print('pick_option: state=', state)
        # print('applicable_options=', applicable_options)
        
        if random.random() < self.epsilon:
            applicable_option_list = []
            for i, op in enumerate(applicable_options):
                if op > 0.1:
                    applicable_option_list.append(i)
            return np.random.choice(applicable_option_list)
        else:
            # TODO: List up available options
            # available_options = XXX
            return self.high_control_target.get_best_action(state_d, applicable_options)[0]
            
    def generate_option(self):
        # TODO: Retrieve samples from the experience buffer.
        #       Train
        # TODO: Implement a control to generate DQN if discrete action space
        op_name = "_op_num" + str(len(self.options))

        option = OptionWrapper(sess=self.sess, obs_dim=self.obs_dim, action_dim=self.action_dim, action_bound=self.action_bound, num_actions=self.num_actions, continuous_action=self.continuous_action, name=self.name + "_inst" + str(self.curr_instances) + op_name)

        return option
        
            
    def get_applicable_options(self, state_d):
        # TODO: Let's first assume all options are always available
        av = np.zeros(self.num_options, dtype=np.float32)
        for i, op in enumerate(self.options):
            if op.is_initiation(state_d):
                av[i] = 1.0
        return av

    def train(self, s, a, r, s2, t, duration, batch_size):
        # TODO: What does this line do?
        targetVals = self.high_control_target.predict_value(s2) # TODO: Do we need self.sess here? why?
        
        y = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if t[i]:
                y[i] = r[i]
            else:
                y[i] = r[i] + math.pow(self.gamma, duration[i]) * targetVals[i]
        loss = self.high_control_main.train(s, a, y)
        print('loss for the main=', loss)

        self.sess.run(self.update_target_params)
        
        return loss

    def train_options(self):
        for op in self.options:
            # TODO: Number of steps for the options needs to be stored
            op.train(self.experience_buffer, self.batch_size)
        
            
    def reset(self):
        # remove all options
        # reset the networks
        self.high_control_main.initialize()
        self.high_control_target.initialize()

        self.option_buffer = ExperienceBuffer(buffer_size=20000)
        
        self.experience_buffer = ExperienceBuffer(buffer_size=100000)
        self.prev_state, self.prev_action = None, None
        self.prev_op_state, self.prev_option = None, None
        self.curr_step, self.total_steps = 0, 0
        self.curr_episodes = 0
        self.total_reward = 0
        
        self.num_op_executed = np.zeros(self.num_options, dtype=np.int32)
        
        primitive_agent = OptionWrapper(sess=self.sess, obs_dim=self.obs_dim, action_dim=self.action_dim, action_bound=self.action_bound, num_actions=self.num_actions, continuous_action=self.continuous_action, name=self.name + "_inst" + str(self.curr_instances) + "_prim")
        
        self.options = []
        self.options.append(primitive_agent)
        self.current_options = None
        self.op_cumulative_reward = 0
        self.op_num_steps = 0
        self.curr_instances += 1
        



class LinearQ():
    def __init__(self, )

class QNetwork():
    """
    Assume the number of actions is discrete.
    """
    def __init__(self, sess, obs_dim=None, num_options=None, learning_rate=0.0001, name="high_ctrl"):
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_options = num_options
        self.obs_dim = obs_dim
        self.name = name
        self.obs, self.q_estm = self.q_network(scope=name + "_q")
        
        self.applicable_options = tf.placeholder(tf.float32, shape=[None, self.num_options], name=name+"_q_applicable_options")

        qmin = tf.reduce_min(self.q_estm, axis=1) - 1.0

        # If applicable_op is True, then return q_estm. Return qmin - 1.0 else.g
        self.applicable_q_value = tf.multiply(self.q_estm, self.applicable_options) + tf.multiply(qmin, tf.add(1.0, -self.applicable_options))
        
        # TODO: How do we make the agent choose the applicable options.
        # self.applicable_q_value = tf.multiply(self.applicable_options, self.q_estm)
        self.best_action = tf.argmax(self.applicable_q_value, 1)

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name=name+"_pred_q")

        self.options = tf.placeholder(tf.int32, [None], name=name+"_options")
        actions_onehot = tf.one_hot(self.options, self.num_options, dtype=tf.float32)
        Q = tf.reduce_sum(actions_onehot * self.q_estm, axis=1)
        # TODO: add entropy loss
        self.loss = tflearn.mean_square(self.predicted_q_value, Q)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

        
        self.network_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name + "_q")
        self.initializer = tf.initializers.variables(self.network_params + self.optimizer.variables())
        

    def q_network(self, scope):
        # TODO: This is probably too deep?
        obs = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name=scope+"_obs")
        # TODO: Should actions be one-hot vector instead of a single variable? I think it should be.
        # actions = tf.placeholder(tf.int32, shape=[None], name=scope+"_actions")
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = tflearn.fully_connected(obs, 400, name='d1', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.obs_dim)))
            net = tf.layers.batch_normalization(net)
            net = tflearn.activations.relu(net)

            net = tflearn.fully_connected(net, 300, name='d2', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/400.0)) # This gives the values from the observation
            net = tflearn.activations.relu(net)

            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out = tflearn.fully_connected(net, self.num_options, weights_init=w_init)

        return obs, out

    def train(self, obs, option, predicted_q_value):
        return self.sess.run([self.optimize, self.loss], feed_dict={
            self.obs: obs,
            self.option: option,
            self.predicted_q_value: predicted_q_value
        })

    def predict_value(self, state):
        vals = self.sess.run(self.q_estm, feed_dict={
            self.obs: state
        })
        return np.max(vals, axis=1)

    def get_best_action(self, obs, applicable_options):
        obs_ = np.reshape(obs, (1, self.obs_dim))
        # print('obs_.shape=', obs_.shape)
        # print('applicable_options.shape=', applicable_options)
        ao_ = np.reshape(applicable_options, (1, self.num_options))
        
        return self.sess.run(self.best_action, feed_dict={
            self.obs: obs_,
            self.applicable_options: ao_
        })

    def initialize(self):
        return self.sess.run(self.initializer, feed_dict={})
