# Python imports.
import tensorflow as tf
import numpy as np
import random
import tflearn
from collections import defaultdict

# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents import DQNAgent, DDPGAgent, LinearQAgent, RandomAgent, RandomContAgent
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer
from simple_rl.agents.func_approx.Features import Fourier

from options.OptionWrapper import OptionWrapper, CoveringOption

class OptionAgent(Agent):
    """
    Components:
    1. DQN to select option
    2. Low level controllers for each option
    3. Spectrum method to Generate options
    """
    NAME = "option-agent"
    
    def __init__(self, sess=None, obs_dim=None, obs_bound=None, action_dim=None, action_bound=None, num_actions=None, num_options=0, gamma=0.99, epsilon=0.0, tau=0.001, high_method='linear', low_method='linear', f_func='fourier', batch_size=32, buffer_size=32, low_update_freq=1, option_batch_size=32, option_buffer_size=32, high_update_freq=10, option_freq=256, option_min_steps=512, init_all=True, init_around_goal=True, init_dist=0.9, term_dist=0.1, bidirectional=False, name=NAME):
        # TODO: Implement an interface for discrete action space
        Agent.__init__(self, name=name, actions=[])

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True # TODO: conv dumps error without this
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess
        self.obs_dim = obs_dim
        self.obs_bound = obs_bound
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.num_actions = num_actions
        # if self.num_actions is None:
        #     self.continuous_action = True
        # else:
        #     self.continuous_action = False
            
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size # TODO: Let's test online learning first.
        self.low_update_freq = low_update_freq
        self.tau = tau
        self.init_around_goal = init_around_goal
        self.init_dist = init_dist
        self.term_dist = term_dist

        # TODO: Should we use this as an initialization process?
        if num_options == 1:
            # Never update the high level policy if there is no options.
            self.high_update_freq = 1000000000000000000
        else:
            self.high_update_freq = high_update_freq
        self.option_batch_size = option_batch_size
        self.option_buffer_size = option_buffer_size # Online setting
        self.option_freq = option_freq
        self.option_min_steps = option_min_steps

        self.num_options = num_options
        self.init_all = init_all
        self.bidirectional = bidirectional

        self.default_options = []

        self.curr_instances = 0
        self.generated_options = dict()

        self.high_method = high_method
        self.low_method = low_method
        self.f_func = f_func
        
        if self.high_method == 'linear':
            # low_bound = np.asarray([0.0, 0.0, -2.0, -2.0])
            # up_bound = np.asarray([1.0, 1.0, 2.0, 2.0])
            features = Fourier(state_dim=obs_dim, bound=obs_bound, order=3)
            self.high_control = LinearQAgent(actions=range(self.num_options), feature=features, name=self.name+"_high")
        elif self.high_method == 'sarsa':
            # low_bound = np.asarray([0.0, 0.0, -2.0, -2.0])
            # up_bound = np.asarray([1.0, 1.0, 2.0, 2.0])
            features = Fourier(state_dim=obs_dim, bound=obs_bound, order=3)
            self.high_control = LinearQAgent(actions=range(self.num_options), feature=features, sarsa=True, name=self.name+"_high")            
        elif self.high_method == 'dqn':
            self.high_control = DQNAgent(sess=self.sess, obs_dim=obs_dim, num_actions=self.num_options, buffer_size=0, gamma=self.gamma, epsilon=self.epsilon, learning_rate=0.001, tau=self.tau, name=self.name+"_high")
        elif self.high_method == 'rand':
            self.high_control = RandomAgent(range(self.num_options), name=self.name+"_high")
        else:
            assert(False)
                    
        self.reset()

    def act(self, state, reward, train=True, data=None):
        # Train the high-level DQN.
        # state_d = state.data.flatten()
        if self.total_steps % self.high_update_freq == 0 and self.option_buffer.size() > self.option_batch_size and train:
            s, a, r, s2, t, duration = self.option_buffer.sample_op(self.option_batch_size)
            # print('exper_buffer.size()=', self.option_buffer.size())
            # print('batchsize=', self.option_batch_size)
            self.high_control.train_batch(s, a, r, s2, t, duration=duration, batch_size=self.option_batch_size)
            
            # print('high_ctrl loss=', loss)

        if self.total_steps > 0 and self.total_steps % self.low_update_freq == 0 and self.experience_buffer.size() > self.batch_size and train:
            # TODO: How freqeuently should you update the options?
            # print('exper_buffer.size()=', self.experience_buffer.size())
            # print('batchsize=', self.batch_size)
            self.train_options()

        # Save sampled transition to the replay buffer
        if not (self.prev_state is None) and not (self.prev_action is None):
            # print('exp buffer added', self.prev_state)
            # print('reward=', reward)
            if data is not None:
                self.experience_buffer.add((self.prev_state, self.prev_action, reward, data, False, self.current_option))
            else:
                self.experience_buffer.add((self.prev_state, self.prev_action, reward, state, state.is_terminal(), self.current_option))

                
        # Generate options
        if self.total_steps % self.option_freq == 0 and self.option_buffer.size() >= self.option_batch_size and self.total_steps >= self.option_min_steps and len(self.options) < self.num_options:
            options = self.generate_option()
            for o in options:
                self.options.append(o)
            print('generated an option')
        # Pick an option
        if self.current_option is None:
            self.current_option = self.pick_option(state)
            self.num_op_executed[self.current_option] += 1
            self.prev_op_state, self.prev_option = state, self.current_option
        else:
            self.op_cumulative_reward = self.op_cumulative_reward + pow(self.gamma, self.op_num_steps) + reward
            self.op_num_steps += 1
            if self.options[self.current_option].is_terminal(state) or state.is_terminal():
                # if state.is_terminal():
                #     print('isterminal')
                # print('picking an option!')
                # Save sampled transition to the replay buffer
                if not (self.prev_op_state is None) and not (self.prev_option is None):
                    # TODO: Discount factor for the Value
                    # print('opt buffer added', self.prev_state)
                    self.option_buffer.add((self.prev_op_state, self.prev_option, self.op_cumulative_reward, state, state.is_terminal(), self.op_num_steps))

                prev_option = self.current_option
                self.current_option = self.pick_option(state)
                
                # if self.options[prev_option].is_terminal(state) and prev_option != 0:
                #     assert(prev_option != self.current_option)
                
                self.num_op_executed[self.current_option] += 1

                self.prev_op_state, self.prev_option = state, self.current_option

                self.op_cumulative_reward = 0
                self.op_num_steps = 0
                
            # else:
                # Contiue on
                # print('option continues!')
                # self.op_cumulative_reward = self.op_cumulative_reward + pow(self.gamma, self.op_num_steps) + reward
                # self.op_num_steps += 1

        # Retrieve an action
        # print('current_option = ', self.current_option)
        # print('#options = ', len(self.options))
        assert(self.current_option < len(self.options))
        
        prim_action = self.options[self.current_option].act(state)


        # print('current_option=', self.current_option, 'action=', prim_action)
        
        self.prev_state, self.prev_action = state, prim_action

        if data is not None:
            self.prev_state = data

        self.curr_step += 1
        self.total_steps += 1

        self.total_reward += reward

        # TODO: when is state is_terminal?
        # if state.is_terminal():
        #     print('#Episode=', self.curr_episodes, '#steps=', self.curr_step, 'Total_reward=', self.total_reward)
        #     print('#Options executed = ', self.num_op_executed)
        #     self.curr_step = 0
        #     self.curr_episodes += 1
        #     self.total_reward = 0
        #     self.current_option = None
        #     self.op_cumulative_reward = 0
        #     self.op_num_steps = 0
            
        
        return prim_action

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        if self.prev_state.is_terminal():
            print('reached the goal')
        print('#Episode=', self.episode_number, '#steps=', self.curr_step, 'Total_reward=', self.total_reward)
        print('#Options executed = ', self.num_op_executed)

        # TODO: Store the transition 

            # if state.is_terminal():
            #     print('isterminal')
            # print('picking an option!')
            # Save sampled transition to the replay buffer
        if not (self.prev_op_state is None) and not (self.prev_option is None) and not (self.prev_state.is_terminal()):
            # TODO: DOes it ignoring the last reward added to the agent?
            self.option_buffer.add((self.prev_op_state, self.prev_option, self.op_cumulative_reward, self.prev_state, self.prev_state.is_terminal(), self.op_num_steps))
                
            self.num_op_executed[self.current_option] += 1
        
        
        self.curr_step = 0
        self.current_option = None
        self.op_cumulative_reward = 0
        self.op_num_steps = 0

        self.prev_state = None
        self.prev_action = None
        self.episode_number += 1


    def pick_option(self, state):
        applicable_option_list = self.get_applicable_options(state)
        assert(len(applicable_option_list) > 0)
        # applicable_option_list = []
        # for i, op in enumerate(applicable_options):
        #     if op > 0.1:
        #         applicable_option_list.append(i)

        # print('pick_option: state=', state)
        # print('applicable_option_list=', applicable_option_list)

        # TODO: Should we induce randomness here?
        if random.random() < self.epsilon:
            return np.random.choice(applicable_option_list)
        else:
            # TODO: List up available options
            # available_options = XXX
            maxqval = float("-inf")
            maxqop = -1
            for o in applicable_option_list:
                assert(type(o) is int)
                val = self.high_control.get_q_value(state, o)
                # print('Q(s,', o, ')=', val)
                # print('type=', type(val))
                if val > maxqval:
                    maxqval = val
                    maxqop = o
            if maxqop == -1:
                for o in applicable_option_list:
                    val = self.high_control.get_q_value(state, o)
                    print('Q(s,', o, ') =', val)
            assert(maxqop >= 0)
            return maxqop
            
    def generate_option(self):
        op_name = "_op_num" + str(len(self.options))
        options = []
        option = CoveringOption(sess=self.sess, experience_buffer=self.option_buffer, option_b_size=self.option_batch_size, obs_dim=self.obs_dim, obs_bound=self.obs_bound, action_dim=self.action_dim, action_bound=self.action_bound, num_actions=self.num_actions, low_method=self.low_method, f_func=self.f_func, init_all=self.init_all, init_around_goal=self.init_around_goal, init_dist=self.init_dist, term_dist=self.term_dist, name='online-option' + str(len(self.options)))
        option.train(experience_buffer=self.experience_buffer, batch_size=self.experience_buffer.size()) # TODO: This may be too large if the buffer size is large.
        options.append(option)
        if self.bidirectional:
            option2 = CoveringOption(sess=self.sess, experience_buffer=self.option_buffer, option_b_size=self.option_batch_size, obs_dim=self.obs_dim, obs_bound=self.obs_bound, action_dim=self.action_dim, action_bound=self.action_bound, num_actions=self.num_actions, low_method=self.low_method, f_func=self.f_func, init_all=self.init_all, reversed_dir=True, init_around_goal=self.init_around_goal, init_dist=self.init_dist, term_dist=self.term_dist, name='online-option' + str(len(self.options)))
            option2.train(experience_buffer=self.experience_buffer)
            options.append(option2)
        return options
            
    def get_applicable_options(self, state):
        l = []
        # av = np.zeros(self.num_options, dtype=np.float32)
        for i, op in enumerate(self.options):
            if op.is_initiation(state):
                l.append(i)
        return l

    # def train(self, s, a, r, s2, t, duration, batch_size):
    #     # TODO: What does this line do?
    #     targetVals = self.high_control_target.predict_value(s2) # TODO: Do we need self.sess here? why?
    #     
    #     y = np.zeros(self.batch_size)
    #     for i in range(self.batch_size):
    #         if t[i]:
    #             y[i] = r[i]
    #         else:
    #             y[i] = r[i] + math.pow(self.gamma, duration[i]) * targetVals[i]
    #     loss = self.high_control_main.train(s, a, y)
    #     print('loss for the main=', loss)
    # 
    #     self.sess.run(self.update_target_params)
    #     
    #     return loss

    def train_options(self):
        for op in self.options:
            # TODO: Number of steps for the options needs to be stored
            op.train(self.experience_buffer, self.batch_size)
        
            
    def reset(self):

        # Save the
        if self.curr_instances > 0:
            self.generated_options[self.curr_instances] = self.options
        
        self.high_control.reset()
        
        self.option_buffer = ExperienceBuffer(buffer_size=self.option_buffer_size)
        
        self.experience_buffer = ExperienceBuffer(buffer_size=self.buffer_size)
        self.prev_state, self.prev_action = None, None
        self.prev_op_state, self.prev_option = None, None
        self.curr_step, self.total_steps = 0, 0
        self.total_reward = 0
        self.episode_number = 0
        
        self.num_op_executed = [0] * self.num_options
        
        primitive_agent = CoveringOption(sess=self.sess, obs_dim=self.obs_dim, obs_bound=self.obs_bound, action_dim=self.action_dim, action_bound=self.action_bound, num_actions=self.num_actions, low_method=self.low_method, f_func=self.f_func, name=self.name + "_inst" + str(self.curr_instances) + "_prim")
        
        
        self.options = []
        self.options.append(primitive_agent)

        # TODO: This doesn't work -- we have to reinitialize the default options on every reset.
        for o in self.default_options:
            self.options.append(o)
        self.current_option = None
        self.op_cumulative_reward = 0
        self.op_num_steps = 0
        self.curr_instances += 1

    def add_option(self, option):
        self.default_options.append(option)
        self.options.append(option)
        assert(len(self.options) <= self.num_options)
        # self.num_op_executed.append(0)

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        
        param_dict["high_method"] = self.high_method
        param_dict["low_method"] = self.low_method
        param_dict["num_options"] = self.num_options

        param_dict["epsilon"] = self.epsilon
        param_dict["gamma"] = self.gamma

        param_dict["low_update_freq"] = self.low_update_freq
        param_dict["batch_size"] = self.batch_size
        param_dict["buffer_size"] = self.buffer_size

        param_dict["high_update_freq"] = self.high_update_freq
        param_dict["option_batch_size"] = self.option_batch_size
        param_dict["option_buffer_size"] = self.option_buffer_size

        param_dict["tau"] = self.tau

        param_dict["init_around_goal"] = int(self.init_around_goal)
        param_dict["init_dist"] = self.init_dist
        param_dict["term_dist"] = self.term_dist
        
        # param_dict["high_params"] = self.high_control.get_parameters()
        # param_dict["low_params"] = self.low_control.get_parameters()
        

        return param_dict
