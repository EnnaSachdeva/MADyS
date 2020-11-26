'''
This is decentralized high level learner, while the low level primitives are centralized across agents
'''

import numpy as no, os, time, sys, random
from core.mod_utils import pprint, str2bool
import torch
from core import buffer
from core.policy_selector_algo import DQN
from torch.multiprocessing import Process, Pipe
import argparse
import numpy as np
import threading, sys
from core.agent import PolicySelectorAgent, PrimitiveAgent, TestAgent
from core.runner import rollout_worker
import core.mod_utils as mod
from core import mod_utils as utils
from core.buffer import Buffer
import pdb
import cProfile

from tensorboardX import SummaryWriter
import datetime
import datetime
#torch.set_num_threads(1) #disable multithreading

# This test is just to test the low level skill learners learning the tasks
render = False
parser = argparse.ArgumentParser()
parser.add_argument('-action_space', type=str, help='different or same?', default='same')  # todo: for different speeds of each agents
parser.add_argument('-config', type=str, help='World Setting?', default='fire_truck_uav')  # todo: change this for different coupling requirements
parser.add_argument('-reward_scheme', type=str, help='Reward Structure? 1. multiple 2. single', default='multiple')
parser.add_argument('-savetag', help='Saved tag', default='')
parser.add_argument('-algo', type=str, help='SAC Vs. TD3?', default='TD3')
parser.add_argument('-evolution_only', type=str, help='Only evolution on low level actions?', default='False')

##### The environmental parameters
parser.add_argument('-num_uav', type=int, help='How many uavs?', default='2')
parser.add_argument('-num_truck', type=int, help='How many trucks?', default='0')

parser.add_argument('-num_poi_A', type=int, help='How many POIs of type A?', default='1')
parser.add_argument('-num_poi_B', type=int, help='How many POIs of type B?', default='1')
parser.add_argument('-num_poi_C', type=int, help='How many POIs of type C?', default='0')
parser.add_argument('-num_poi_D', type=int, help='How many POIs of type D?', default='0')

parser.add_argument('-poi_sequence', type=int, help='What sequence to follow?', default={0: None, 1: [0]})

parser.add_argument('-coupling_uav', type=int, help='How many uavs for POI?', default='2')
parser.add_argument('-coupling_truck', type=int, help='How many trucks for POI?', default='0')

# seed
parser.add_argument('-seed', type=int, help='which seed?', default='1')

RANDOM_BASELINE_DQN = False # kept true in new data
RANDOM_BASELINE_TD3 = False# False
PRIMITIVE_ROLLOUTS = False # for primitive parallel rollouts
CARDINAL_ACTIONS_ALLOWED = False
LOW_PRETRAINED = False #uses the pre-trained low level policies


###### Profile decorator #####
import cProfile, pstats, io
def profile(fnc):
    def inner (*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner


class ConfigSettings:
    def __init__(self):
        ### Environment config settings
        self.action_space = vars(parser.parse_args())['action_space']
        self.config = vars(parser.parse_args())['config']
        self.reward_scheme = vars(parser.parse_args())['reward_scheme']
        self.evolution_only = vars(parser.parse_args())['evolution_only']

        #### Environment configs
        self.num_uav = vars(parser.parse_args())['num_uav']
        self.num_truck = vars(parser.parse_args())['num_truck']
        self.num_poi_A = vars(parser.parse_args())['num_poi_A']
        self.num_poi_B = vars(parser.parse_args())['num_poi_B']
        self.num_poi_C = vars(parser.parse_args())['num_poi_C']
        self.num_poi_D = vars(parser.parse_args())['num_poi_D']

        self.poi_sequence = vars(parser.parse_args())['poi_sequence']

        self.coupling_uav = vars(parser.parse_args())['coupling_uav']
        self.coupling_truck = vars(parser.parse_args())['coupling_truck']

        ## seed
        self.seed = vars(parser.parse_args())['seed']

        # eventually need to remove the below 2 parameters
        #self.is_proxim_rew = True
        #self.is_gsl = False

        #if self.config == 'fire_truck_uav':  # just one resolution angle has a long range, others are short
        # Rover domain
        self.dim_x = self.dim_y = 20;
        # self.obs_radius = self.dim_x * 10;
        self.act_dist = 2;
        self.rover_speed = 1;
        self.sensor_model = 'closest'
        self.angle_res = 90
        self.num_poi = self.num_poi_A + self.num_poi_B + self.num_poi_C + self.num_poi_D

        self.harvest_period = 1
        self.num_poi_types = 0
        self.num_uavs = self.num_uav
        self.num_trucks = self.num_truck

        # self.num_agents = self.num_agent_types * self.num_agents_per_type
        self.num_agents = self.num_uavs + self.num_trucks

        self.percentage_uav = 800
        self.percentage_truck = 800

        self.long_range = np.sqrt((self.percentage_truck / (100 * 3.14))) * self.dim_x

        self.ep_len = 70
        self.poi_rand = 1
        # self.coupling = 2
        obs = np.sqrt((self.percentage_uav / (100 * 3.14))) * self.dim_x # for UAV

        #self.env_choice = 'rover_homogeneous' # todo: need to change this
        self.env_choice = 'rover_heterogeneous' # todo: need to change this

        ##### For POI types
        self.poi_types = []
        self.reward_types = {}
        reward_idx = -1

        if self.num_poi_A > 0:
            reward_idx +=1
            self.num_poi_types += 1
            pois = [0]*self.num_poi_A
            self.poi_types.extend(pois)
            self.reward_types[reward_idx] = 'go_to_POI_A'
            reward_idx += 1
            self.reward_types[reward_idx] = 'go_away_from_POI_A'

        if self.num_poi_B > 0:
            reward_idx += 1
            self.num_poi_types += 1
            pois = [1] * self.num_poi_B
            self.poi_types.extend(pois)
            self.reward_types[reward_idx] = 'go_to_POI_B'
            reward_idx += 1
            self.reward_types[reward_idx] = 'go_away_from_POI_B'

        if self.num_poi_C > 0:
            reward_idx += 1
            self.num_poi_types += 1
            pois = [2] * self.num_poi_C
            self.poi_types.extend(pois)
            self.reward_types[reward_idx] = 'go_to_POI_C'
            reward_idx +=1
            self.reward_types[reward_idx] = 'go_away_from_POI_C'

        if self.num_poi_D > 0:
            reward_idx += 1
            self.num_poi_types += 1
            pois = [3] * self.num_poi_D
            self.poi_types.extend(pois)
            self.reward_types[reward_idx] = 'go_to_POI_D'
            reward_idx += 1
            self.reward_types[reward_idx] = 'go_away_from_POI_D'

        ##### For agents types
        self.obs_radius = []
        self.coupling = []
        if self.num_uavs > 0:
            # number of primitive types
            primitive_types = 1
            self.num_agent_types = 1  # for firetruck and UAVs (type: 0 for UAV and type: 1 for firetruck)
            self.obs_radius.append(np.sqrt((self.percentage_uav / (100 * 3.14))) * self.dim_x)  # for UAV
            self.coupling.append(self.coupling_uav)  # for UAV
            if self.num_uavs > 1:
                reward_idx += 1
                self.reward_types[reward_idx] = 'go_to_uav'
                reward_idx += 1
                self.reward_types[reward_idx] = 'go_away_from_uav'


        if (self.num_trucks > 0):
            primitive_types += 1
            self.num_agent_types += 1  # for different percentage obs
            self.env_choice = 'rover_heterogeneous'
            self.obs_radius.append(np.sqrt((self.percentage_truck / (100 * 3.14))) * self.dim_x)  # for truck
            self.coupling.append(self.coupling_truck)  # for truck
            # obs.append(2 * self.dim_x)  # for UAV
            # obs.append(np.sqrt(2)*self.dim_x/2) # for fire truck
            # for i in range(self.num_agent_types):
            #	obs.append(2*self.dim_x * 10/(i+1))
            if self.num_uavs > 1:
                reward_idx += 1
                self.reward_types[reward_idx] = 'go_to_truck'
                reward_idx += 1
                self.reward_types[reward_idx] = 'go_away_from_truck'


        self.EVALUATE_only = False # currently

        self.low_pretrained = LOW_PRETRAINED # currently

        print("Configuration:", "Agent Types- ", self.num_agent_types, "\n", "Number of UAVs- ", self.num_uavs, "\n", "Number of Trucks- ",  self.num_trucks, "\n", "POI types- ", self.num_poi_types,
              "\n", "POI A- ", self.num_poi_A, "\n", "POI B- ", self.num_poi_B, "\n", "POI C- ", self.num_poi_C, "\n", "POI D- ", self.num_poi_D)

        print()

        print("Reward Types:", self.reward_types)

################################################################################
####### Parameters for DQN, TD3, Evolutionary reinforcement learning ##########
################################################################################

class Parameters:
    def __init__(self):
        self.num_frames = 100000000
        #self.num_frames = 1000

        self.config = ConfigSettings()
        self.independent_primitive_rollouts = PRIMITIVE_ROLLOUTS

        if CARDINAL_ACTIONS_ALLOWED:
            self.cardinal_actions = 4
        else:
            self.cardinal_actions = 0

        self.actualize = False
        self.priority_rate = 0.0

        # FOR CUDA
        self.is_cuda = True
        self.is_memory_cuda = True
        self.use_gpu = torch.cuda.is_available()

        # Synchronization period, from PG part of DQN to the Evo part
        self.sync_period = 2

        # DQN parameters
        self.use_ln = True
        self.tau = 0.001
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.learning_rate = 1e-4
        self.target_update_interval = 5
        self.dqn_target_update_interval = 2000 # todo: need to change it prolly
        self.DQN_buffer_capacity = 10000

        # TD3 Multiagent Policy Gradient Algorithm
        self.hidden_size = 256
        self.algo_name = vars(parser.parse_args())['algo']
        self.actor_lr = 5e-5 #5e-3f
        self.critic_lr =1e-5 #1e-4
        self.init_w = True
        self.gradperstep = 0.6
        #self.gamma = 0.5 if self.policy_selector_popn_size > 0 else 0.97
        self.gamma = 0.5
        self.TD3_batch_size = 512#256
        self.TD3_buffer_capacity = 50000
        # self.buffer_size = 10
        self.filter_c = 1
        self.reward_scaling = 10.0  # use it in DQN while updating the params
        self.action_loss = False
        self.policy_ups_freq = 2
        self.policy_noise = True
        self.policy_noise_clip = 0.4

        # DQN parameters
        self.DQN_batch_size = 256
        self.epsilon = 0.0

        # test run params
        self.policy_selector_popn_size = 40
        self.policy_selector_rollout_size = 0
        self.overall_num_test = 5

        self.primitives_gen = 5
        self.primitives_rollout_size = 10  # changed from 20
        self.primitives_test_rollouts = 5  # changed from 10

        self.test_gen = 5
        self.tensorboard_gen = self.test_gen

        # Neuroevolution params
        self.num_evals = 1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnitude = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_clamp = 1000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
        self.lineage_depth = 10
        self.ccea_reduction = "lineancy"
        self.num_anchors = 5

        self.elite_fraction = 0.4 # what part of the total pop should be chosen as elites

        self.num_elites = int(self.elite_fraction * self.policy_selector_popn_size)
        self.num_blends = int(0.15 * self.policy_selector_popn_size)

        #self.state_dim = int(360*(self.config.num_agent_types+self.config.num_poi_types) / self.config.angle_res) + 1
        #self.state_dim += 2 # for velocity, an additional
        self.state_dim = int(360*(self.config.num_agent_types + self.config.num_poi_types) / self.config.angle_res)

        self.agent_action_dim = 2 # for <dx, dy>
        self.policy_selector_action_dim = len(self.config.reward_types)   # for <dx, dy>

        self.algo_name = vars(parser.parse_args())['algo']

        ############# Creating folders to save the models
        # Save Filenames
        self.savetag = vars(parser.parse_args())['savetag'] + \
                       'pop' + str(self.policy_selector_popn_size) + \
                       '_roll' + str(self.policy_selector_rollout_size) + \
                       '_env-' + str(self.config.env_choice) + '_' + str(self.config.config) + \
                       '_action_' + str(self.config.action_space) + \
                       '_seed' + str(self.config.seed) + \
                       '_reward' + str(self.config.reward_scheme) + \
                       '_uav' + str(self.config.num_uavs) + \
                       '_trucks' + str(self.config.num_trucks) + \
                       '_coupling-uav' + str(self.config.coupling_uav) + \
                       '_coupling-truck' + str(self.config.coupling_truck) + \
                       '_obs-uav' + str(self.config.percentage_uav) + \
                       '_obs-truck' + str(self.config.percentage_truck)

        # save result
        self.save_foldername = 'MADS_' + str(self.config.num_uavs)+ '_' + str(self.config.coupling_uav) + '_' + str(self.config.num_poi_types ) + '_' + str(self.config.ep_len) + '_' + str(self.config.seed) + '/'
        #self.save_foldername = '/nfs/hpc/share/solankis/Homogeneous_learning_from_whole_buffer_random_new' + str(self.config.seed) + '/'

        self.metric_save = self.save_foldername + 'metrics/'
        self.primitive_model_save = self.save_foldername + 'primitive_models/'
        self.policy_selector_q_save = self.save_foldername + 'policy_selector_q_save/'
        self.policy_selector_champ_save = self.save_foldername + 'policy_selector_evo_champ_save/'
        self.tensorboard_save = self.save_foldername + 'tensorbaord/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if not os.path.exists(self.save_foldername):os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.primitive_model_save): os.makedirs(self.primitive_model_save)
        if not os.path.exists(self.policy_selector_q_save): os.makedirs(self.policy_selector_q_save)
        if not os.path.exists(self.policy_selector_champ_save): os.makedirs(self.policy_selector_champ_save)

        self.critic_fname = 'critic_' + self.savetag
        self.actor_fname = 'actor_' + self.savetag
        self.log_fname = 'reward_' + self.savetag
        self.best_fname = 'best_' + self.savetag

        ######## TensorBoard plots
        self.writer = SummaryWriter(log_dir=self.tensorboard_save)

################################################################################
####### Heirarchical multi-reward evolutionary reinforcement learning ##########
################################################################################

class H_MRERL:
    def __init__(self, args):
        self.args = args;

        ####################### Primitive agents (PG learner) ##########################

        self.primitive_agents = []
        self.primitive_agents_rollout_bucket = []

        #from torch.multiprocessing import Manager
        #manager = Manager()
        #self.primitive_agents_rollout_bucket = manager.dict()

        for reward_type_idx, reward_type in self.args.config.reward_types.items():
            #self.primitive_agents[reward_type] = PrimitiveAgent(self.args, 0)
            #self.primitive_agents_rollout_bucket[reward_type] = self.primitive_agents[reward_type].rollout_actor
            self.primitive_agents.append(PrimitiveAgent(self.args, reward_type_idx))
            self.primitive_agents_rollout_bucket.append(self.primitive_agents[reward_type_idx].rollout_actor)

        #################### Policy Selector ERL agents #######################
        if self.args.policy_selector_popn_size > 0:
            self.pop = []
            for _ in range(self.args.policy_selector_popn_size):
                self.pop.append(DQN(self.args))  # Centralized policy selector

        self.policy_selector_agent = [PolicySelectorAgent(self.args, id) for id in range(self.args.config.num_agents)] # decentralized agents

        ################ Different buffer for each agent #######################
        '''
        buffer = Buffer(self.args.TD3_buffer_capacity, buffer_gpu=False, buffer_type='Policy_selector', filter_c=self.args.filter_c)  # this is for one agent

        self.centralized_buffer = [buffer for _ in range(self.args.config.num_agents)]
        self.centralized_buffer_dict = [buffer.tuples for buffer in self.centralized_buffer]
        '''

        ### Shared buffer
        buffer = Buffer(self.args.TD3_buffer_capacity, buffer_gpu=False, buffer_type='Policy_selector', filter_c=self.args.filter_c)  # this is for one agent
        self.centralized_buffer = [buffer] # for shared
        self.centralized_buffer_dict = [buffer.tuples for buffer in self.centralized_buffer]
        self.centralized_buffer = self.centralized_buffer[0]


        #################### Test Agent ########################
        self.test_agent = TestAgent(self.args)

        ######### Initialize networks for multiagent PG, Policy selector PG, Policy selector evo #######
        self.popn_bucket = [ag.popn for ag in self.policy_selector_agent]
        self.policy_selector_rollout_bucket = [ag.rollout_actor for ag in self.policy_selector_agent]
        self.test_bucket = self.test_agent.rollout_actor

        # Primitive Workers
        if self.args.primitives_rollout_size and self.args.independent_primitive_rollouts and (not self.args.config.low_pretrained):

            self.primitive_agents_task_pipes = {}
            self.primitive_agents_result_pipes = {}

            for _, reward_type in self.args.config.reward_types.items():
                self.primitive_agents_task_pipes[reward_type] = Pipe()  # for each reward type
                self.primitive_agents_result_pipes[reward_type] = Pipe()  # for each reward type

            # Create process for each reward
            # todo: no primitive rollouts needed
            self.primitive_agents_workers = [Process(target=rollout_worker, args=(self.args, reward_type_idx, 'primitive_policies', # args, id, type,
                                                                                  self.primitive_agents_task_pipes[reward_type][1], # task_pipe
                                                                                  self.primitive_agents_result_pipes[reward_type][0],# result_pipe
                                                                                  self.centralized_buffer_dict , # policy selector buffer bucket
                                                                                  self.primitive_agents_rollout_bucket[reward_type_idx],# models_bucket
                                                                                  self.policy_selector_rollout_bucket,  # primitive_policies_bucket for all reward types
                                                                                  True,# todo: change this store_transitions
                                                                                  RANDOM_BASELINE_DQN,                      #random_baseliness
                                                                                  RANDOM_BASELINE_TD3,
                                                                                  0.0,
                                                                                  reward_type_idx)) for reward_type_idx, reward_type in self.args.config.reward_types.items()]  # epsilon
            # starting each process, this will happen parallely
            for worker in self.primitive_agents_workers: worker.start()


        ######## Policy Selector- Evolutionary workers: different processes for each member of the popn
        if self.args.policy_selector_popn_size > 0:
            self.policy_selector_evo_task_pipes = [Pipe() for _ in range(args.policy_selector_popn_size * args.num_evals)]
            self.policy_selector_evo_result_pipes = [Pipe() for _ in range(self.args.policy_selector_popn_size * self.args.num_evals)]
            self.policy_selector_evo_workers =[Process(target=rollout_worker, args=(self.args, i, 'policy_selector_evo',#args, id, type,
                                                                                    self.policy_selector_evo_task_pipes[i][1], # task_pipe
                                                                                    self.policy_selector_evo_result_pipes[i][0],# result_pipe
                                                                                    self.centralized_buffer_dict,
                                                                                    self.primitive_agents_rollout_bucket,# models_bucket
                                                                                    self.popn_bucket,               # primitive_policies_for reward types
                                                                                    self.args.policy_selector_popn_size > 0,                                # store_transitions
                                                                                    RANDOM_BASELINE_DQN,                     # random_baseliness
                                                                                    RANDOM_BASELINE_TD3,
                                                                                    0.0,
                                                                                    -1))                               # reward_type_idx
                                                                                    for i in range(self.args.policy_selector_popn_size * self.args.num_evals)]

            for worker in self.policy_selector_evo_workers: worker.start();

        ############ DQN workers: One process for PG worker
        if self.args.policy_selector_rollout_size > 0:
            self.policy_selector_pg_task_pipes = Pipe()
            self.policy_selector_pg_result_pipes = Pipe()
            self.policy_selector_pg_workers= [Process(target=rollout_worker, args=(self.args, 0, 'policy_selector_pg',     #args, id, type,
                                                                                   self.policy_selector_pg_task_pipes[1],  # task_pipe
                                                                                   self.policy_selector_pg_result_pipes[0],# result_pipe
                                                                                   self.centralized_buffer_dict,  # giving the whole buffer, for all reward types
                                                                                   self.primitive_agents_rollout_bucket,     # sending all primiteve agents bucket
                                                                                   self.policy_selector_rollout_bucket,    # models_bucket
                                                                                   self.args.policy_selector_rollout_size > 0,             # store_transitions
                                                                                   RANDOM_BASELINE_DQN,                        # random_baseliness
                                                                                   RANDOM_BASELINE_TD3,
                                                                                   self.args.epsilon,
                                                                                   -1))] # epsilon                                 # reward_type_idx

            for worker in self.policy_selector_pg_workers: worker.start()

        ######### Test Workers: one process started ############
        if self.args.overall_num_test > 0:
            self.test_task_pipes = Pipe()
            self.test_result_pipes = Pipe()

            self.test_workers = [Process(target=rollout_worker, args=(self.args, 0, 'test',  # args, id, type,
                                                                     self.test_task_pipes[1],  # task_pipe
                                                                     self.test_result_pipes[0],  # result_pipe
                                                                     None,  # centralized Buffer
                                                                     self.primitive_agents_rollout_bucket,# primitive policies for all reward types
                                                                     self.test_bucket,  # models_bucket
                                                                     False,  # store_transitions
                                                                     RANDOM_BASELINE_DQN,  # random_baseliness
                                                                     RANDOM_BASELINE_TD3,
                                                                     0.0,
                                                                    -1))]        # for complete test


            for worker in self.test_workers: worker.start()

        ######### Only Primitive test Workers: one process started ############
        if self.args.primitives_test_rollouts > 0  and (not self.args.config.low_pretrained) :

            self.primitives_test_task_pipes = {}
            self.primitives_test_result_pipes = {}

            for _, reward_type in self.args.config.reward_types.items():
                self.primitives_test_task_pipes[reward_type] = Pipe()  # for each reward type
                self.primitives_test_result_pipes[reward_type] = Pipe()  # for each reward type

            self.primitives_test_workers = [Process(target=rollout_worker, args=(self.args, reward_type_idx, 'primitive_test', # args, id, type,
                                                                                  self.primitives_test_task_pipes[reward_type][1], # task_pipe
                                                                                  self.primitives_test_result_pipes[reward_type][0],# result_pipe
                                                                                  None, #centralized buffer
                                                                                  self.primitive_agents_rollout_bucket[reward_type_idx],#  policy selector buffer bucket
                                                                                  None,   # primitive_policies_bucket for all reward types
                                                                                  False,# store_transitions
                                                                                  RANDOM_BASELINE_DQN,                      #random_baseliness
                                                                                  RANDOM_BASELINE_TD3,
                                                                                  0.0,
                                                                                  reward_type_idx)) for reward_type_idx, reward_type in self.args.config.reward_types.items()]  # epsilon
            for worker in self.primitives_test_workers: worker.start()

        ########### Stats and Tracking of which rollout is done
        self.best_score = -999;
        self.total_frames = 0;
        self.gen_frames = 0;
        self.test_trace = [];


    def make_teams(self, num_agents, popn_size, num_evals):
        temp_inds = []
        for _ in range(num_evals): temp_inds += list(range(popn_size))  # this gives [0,1,2,3... (pop_size-1), 0,1,2,3.., (pop_size-1), for num_evals times]
        all_inds = [temp_inds[:] for _ in range(num_agents)] # repeating the above list for all agents [[0,1,2,3...], [0,1,2,3,...]..[]]
        for entry in all_inds: random.shuffle(entry)
        teams = [[entry[i] for entry in all_inds] for i in range(popn_size * num_evals)] # number of teams = pop_size * number of evals
        return teams

       ################################################################################
       #################### Multiagent Policy Gradient Learning #######################
       ################################################################################

    #@profile
    def train(self, gen, test_tracker, epsilon):
        self.gen_frames = 0

       ################ Start threads across all the processes starte #################
       ################ Threads may be running on different processors ################
       ################ but they will be running one at a time ########################

       ############# Start Test rollouts
        if gen % self.args.test_gen == 0 and self.args.overall_num_test > 0:
            self.test_agent.get_champ(self.policy_selector_agent)  # Sync the champ policies into the TestAgent
            self.test_task_pipes[0].send(["START", None])  # sending START signal

       ############ Start Test Primitive rollouts
        if gen % self.args.test_gen == 0 and self.args.primitives_test_rollouts > 0 and (not self.args.config.low_pretrained):
            for reward_id, reward_type in self.args.config.reward_types.items():
                self.primitives_test_task_pipes[reward_type][0].send(['START', None])  # todo: just to see episodic return

       ############### Policy Selector: ERL ###################
       ###### Start Evo rollouts
        if self.args.policy_selector_popn_size > 0:
            #teams = [[i] for i in list(range(self.args.policy_selector_popn_size))] # todo: used this trick
            teams = self.make_teams(args.config.num_agents, args.policy_selector_popn_size, args.num_evals)  # Heterogeneous Case
            for pipe, team in zip(self.policy_selector_evo_task_pipes, teams):
                pipe[0].send([team, None]) # sending agent signal, as in which agent has to take its rollout now

        ####### Start DQN rollouts
        if self.args.policy_selector_rollout_size > 0:
            #synch pg_actors to its corresponding rollout_bucket
            for agent in self.policy_selector_agent: agent.update_rollout_actor() # copy DQN rained model weights
            self.policy_selector_pg_task_pipes[0].send(['START', epsilon])

        ############# Start Primitive agents rollouts
        # if (gen% self.args.primitives_gen ==0) and self.args.primitives_rollout_size > 0:
        if self.total_frames < 1000000 and self.args.primitives_rollout_size > 0 and (not self.args.config.low_pretrained): #todo: change this
        #if self.args.primitives_rollout_size > 0 and (not self.args.config.low_pretrained): #todo: change this
            # synch pg_actors to its corresponding rollout_bucket
            for reward_id, reward_type in self.args.config.reward_types.items():
                self.primitive_agents[reward_id].update_rollout_actor()  # copy trained weights

            if self.args.independent_primitive_rollouts : #todo: change this
                for reward_id, reward_type in self.args.config.reward_types.items():
                    self.primitive_agents_task_pipes[reward_type][0].send(['START', None])

        ############### Receieve the data from the corresponding rollouts ###################
        episodic_reward = {}
        #if self.args.primitives_rollout_size > 0 and self.args.independent_primitive_rollouts and (not self.args.config.low_pretrained):
        if self.total_frames < 1000000 and self.args.primitives_rollout_size > 0 and self.args.independent_primitive_rollouts and (not self.args.config.low_pretrained):

            for _, reward_type in self.args.config.reward_types.items():
                entry = self.primitive_agents_result_pipes[reward_type][1].recv()
                team = entry[0]
                fitness = entry[1][0]

                episodic_reward[reward_type] = []
                local_reward = []
                for i in range(self.args.config.num_agents):
                    #local_reward = entry[2][0][i].tolist()
                    #local_reward.append(entry[2][0][i])
                    episodic_reward[reward_type].append(utils.list_mean(entry[2][0][i]))

                # don't add frames for primitives
                frames = entry[3] # todo: remove this for overall learning
                self.total_frames += frames

        ## Policy selector evo rollouts
        all_fits = []
        if self.args.policy_selector_popn_size > 0:
            #for pop_id, pipe in enumerate(self.policy_selector_evo_result_pipes): # each agent denotes different neural network
            for pipe in self.policy_selector_evo_result_pipes:  # each agent denotes different neural network
                entry = pipe[1].recv()
                team = entry[0]
                fitness = entry[1][0]
                frames = entry[3]
                #rollout_trajectory = entry[4]
                #its not slowing down, checked it
                #[self.centralized_buffer[agent_id].add(rollout_trajectory[agent_id]) for agent_id in range(self.args.config.num_agents)]

                for agent_id, popn_id in enumerate(team):
                    self.policy_selector_agent[agent_id].fitness[popn_id].append(utils.list_mean(fitness))
                all_fits.append(utils.list_mean(fitness))
                self.total_frames += frames

        if self.args.primitives_rollout_size > 0 and (not self.args.config.low_pretrained): # don;t update if pretrained policies are used

            threads = [threading.Thread(target=self.primitive_agents[reward_id].update_parameters,
                                        args=(self.centralized_buffer, reward_id,))
                       for reward_id, reward_type in self.args.config.reward_types.items()]

            [threads[reward_idx].start() for reward_idx, _ in self.args.config.reward_types.items()]  # start threads
            [threads[reward_idx].join() for reward_idx, _ in self.args.config.reward_types.items()]  # start threads

        ## Policy selector DQN rollouts
        policy_selector_fits = []
        if self.args.policy_selector_rollout_size > 0:
            entry = self.policy_selector_pg_result_pipes[1].recv()
            policy_selector_fits = entry[1][0]
            self.total_frames += entry[3]
            #rollout_trajectory = entry[4]
            #its not slowing down, checked it
            #[self.centralized_buffer[agent_id].add(rollout_trajectory[agent_id]) for agent_id in range(self.args.config.num_agents)]

            # Start PG updates
            threads = [threading.Thread(target=agent.update_parameters, args=(self.centralized_buffer[agent_id], )) for agent_id, agent in enumerate(self.policy_selector_agent)]
            # start threads
            for thread in threads: thread.start()

            # Join threads
            for thread in threads: thread.join()

        ## Test rollouts
        test_fits = []
        rewards_selected = []
        if gen % self.args.test_gen == 0 and self.args.overall_num_test > 0:

            entry = self.test_result_pipes[1].recv()
            test_fits = entry[1][0]
            rewards_selected = entry[3]

            test_tracker.update([mod.list_mean(test_fits)], self.total_frames)
            #test_tracker.update([mod.list_mean(test_fits)], self.total_frames)
            self.test_trace.append(mod.list_mean(test_fits))

        ## Primitives Test rollouts
        if (gen % self.args.test_gen ==0) and self.args.primitives_test_rollouts > 0 and (not self.args.config.low_pretrained):

            for _, reward_type in self.args.config.reward_types.items():
                entry = self.primitives_test_result_pipes[reward_type][1].recv()
                team = entry[0]
                fitness = entry[1][0]

                episodic_reward[reward_type] = []
                local_reward = []
                for i in range(self.args.config.num_agents):
                    #local_reward = entry[2][0][i].tolist()
                    #local_reward.append(entry[2][0][i])
                    episodic_reward[reward_type].append(utils.list_mean(entry[2][0][i]))

                # don't add frames for primitives
                #frames = entry[3]
                #self.total_frames += frames

        ## Policy selector evo rollouts

        ###### Evolution step
        if self.args.policy_selector_popn_size > 0:
            for agent in self.policy_selector_agent:
                agent.evolve()

        ######################## Saving model #########################
        if gen % 1 == 0:

            ######### save primitive models ########
            if (not self.args.config.low_pretrained) and len(self.args.config.reward_types) > 0: # save only if pretained policies are not being used
                for reward_type_idx, reward_type in self.args.config.reward_types.items():
                    for temp_actor in self.primitive_agents[reward_type_idx].rollout_actor:
                        torch.save(temp_actor.state_dict(), self.args.primitive_model_save + reward_type + '_' + self.args.actor_fname + 'primitive')

            ######### save DQN model model ########
            for id, policy_selector_actor in enumerate(self.policy_selector_agent):
                for temp_actor in policy_selector_actor.rollout_actor:
                    torch.save(temp_actor.state_dict(), self.args.policy_selector_q_save + str(id) + '_' + self.args.actor_fname)

            ######### save champion model ########
            for id, test_actor in enumerate(self.test_agent.rollout_actor):
                torch.save(test_actor.state_dict(), self.args.policy_selector_champ_save + str(id) + '_' + self.args.actor_fname)

            #print('Models Saved')

        return all_fits, episodic_reward, policy_selector_fits, test_fits, rewards_selected # todo: return test_fitness too, test_fits



if __name__ == "__main__":
    args = Parameters()
    test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')  # Initiate tracker

    # seed
    torch.manual_seed(args.config.seed)
    np.random.seed(args.config.seed)
    random.seed(args.config.seed)

    # Create Agent, main agent inside which both PG and ERL operations happen
    arch = H_MRERL(args)

    print('Running ', args.config.env_choice, 'with config ', args.config.config, ' State_dim:', args.state_dim,
          'Agents Action_dim', args.agent_action_dim, 'Policy Selector Action_dim', args.policy_selector_action_dim)
    print("Trucks: ", args.config.num_trucks)
    print("UAVs: ", args.config.num_uavs)
    print("POIs: ", args.config.num_poi)

    time_start = time.time()

    epsilon = 0.9999
    #args.epsilon = epsilon
    total_frames_now = 0
    ###### TRAINING LOOP ########
    for gen in range(1, 10000000000):
    #for gen in range(1, 3):
        #if(arch.total_frames - total_frames_now) > 20000:
            #print(arch.total_frames, total_fsrames_now)
        if gen % 2 == 0:
            epsilon = epsilon * 0.99
            if epsilon < 0.01: epsilon = 0.01

            #arch.epsilon = epsilon

        total_frames_now =  arch.total_frames

        popn_fits, episodic_reward, policy_selector_fits, test_fits, rewards_selected = arch.train(gen, test_tracker, epsilon)

        #print("Buffer length: ", arch.centralized_buffer[0].__len__() )
        print("Buffer length: ", arch.centralized_buffer.__len__())

    # PRINT PROGRESS
        if args.policy_selector_popn_size>0: # only if there is an evo part
            print('Gen:/Frames', gen, '/', arch.total_frames, 'Popn stat:', mod.list_stat(popn_fits), 'DQN_stat:',
                  mod.list_stat(policy_selector_fits), 'Test_trace:', [pprint(i) for i in arch.test_trace[-5:]], 'FPS:',
                  pprint(arch.total_frames / (time.time() - time_start)))  # we print the min, max, mean and std deviation of all the values
        else:
            print('Gen:/Frames', gen, '/', arch.total_frames, 'Test_trace:', episodic_reward, 'FPS:',
                  pprint(arch.total_frames / (time.time() - time_start)))  # we print the min, max, mean and std deviation of all the values

        if args.policy_selector_popn_size >0 and gen % args.tensorboard_gen == 0:
            print("Number of times each reward is selected: ")
            # todo: uncomment this
            for k in range(len(rewards_selected)):
                print("Agent-", k, rewards_selected[k])
            args.writer.add_scalar('Average Global Reward of Policy Selector (Champion)', np.mean(test_fits), arch.total_frames)

            if arch.policy_selector_agent[0].algo.q_loss['mean'] != None:
                args.writer.add_scalar('Average fitness of Policy Selector Population', np.mean(popn_fits), arch.total_frames)
                args.writer.add_scalar('Frames/second', arch.total_frames / (time.time() - time_start), arch.total_frames)
                args.writer.add_scalar('Epsilon', epsilon, arch.total_frames)

            ### DQN parameters
            # DQN network weights
            if args.policy_selector_rollout_size > 0:
                args.writer.add_histogram('DQN last layer weights', arch.policy_selector_agent[0].algo.policy.linear3.weight, arch.total_frames)

                for agent_id in range(args.config.num_agents):
                    #data = {}
                    if agent_id < args.config.num_uavs: m = 'UAV'
                    else: m = 'Truck'
                    args.writer.add_scalars('DQN params_' + m + '_' + str(agent_id), {'Q value': arch.policy_selector_agent[agent_id].algo.q_value['mean'],
                                                          'Target Q value': arch.policy_selector_agent[agent_id].algo.target_q_value['mean'],
                                                          'Policy loss': arch.policy_selector_agent[agent_id].algo.q_loss['mean']}, arch.total_frames)

        if args.primitives_test_rollouts > 0 and gen % args.tensorboard_gen == 0 and arch.primitive_agents[0].algo.policy_loss['mean'] != None and (not args.config.low_pretrained):
            for reward_type_idx, reward_type in args.config.reward_types.items():
                args.writer.add_scalars(reward_type, {'Q value': arch.primitive_agents[reward_type_idx].algo.q['mean'],
                                                      'loss': arch.primitive_agents[reward_type_idx].algo.policy_loss['mean']}, arch.total_frames)

                # visualize histogram of weights
                args.writer.add_histogram(reward_type, arch.primitive_agents[reward_type_idx].algo.policy.linear2.weight,
                                      arch.total_frames)

        if args.primitives_gen > 0 and gen % args.tensorboard_gen == 0 and arch.primitive_agents[0].algo.policy_loss['mean'] != None and (not args.config.low_pretrained):

            for agent_id in range(args.config.num_agents):
                data = {}
                for _, reward_type in args.config.reward_types.items():
                    data.update({reward_type: episodic_reward[reward_type][agent_id]})
                if agent_id < args.config.num_uavs: m = 'UAV'
                else: m = 'Truck'
                args.writer.add_scalars('Episodic return_' + m + '_' + str(agent_id), data, arch.total_frames)


        if gen >2 and gen % args.test_gen == 0 and args.primitives_rollout_size > 0 and arch.primitive_agents[0].algo.policy_loss['mean'] != None and (not args.config.low_pretrained):
            print()
            print('Test_stats- Reward used by agents in each time step- ')
            print(mod.list_stat(test_fits))

            for reward_type_idx, reward_type in args.config.reward_types.items():
                print("Reward:", reward_type)
                print("Q", pprint(arch.primitive_agents[reward_type_idx].algo.q))
                print("Policy loss", pprint(arch.primitive_agents[reward_type_idx].algo.policy_loss))
                print()

            print("Primitive Agents performance: Episodic return")

            for _, reward_type in args.config.reward_types.items():
                print(reward_type, episodic_reward[reward_type])

        if gen >2 and gen % (2*args.test_gen) == 0 and args.policy_selector_rollout_size > 0 and arch.policy_selector_agent[0].algo.q_loss['mean'] != None:
            print("DQN:")
            print("Loss", pprint(arch.policy_selector_agent[0].algo.q_loss))
            print("Q value", pprint(arch.policy_selector_agent[0].algo.q_value))
            print("Q target", pprint(arch.policy_selector_agent[0].algo.target_q_value))

        if arch.total_frames >= args.num_frames:
            args.writer.close()

            break

    ###Kill all processes
    try: arch.policy_selector_pg_task_pipes[0].send(['TERMINATE', None])
    except: None
    try: arch.test_task_pipes[0].send(['TERMINATE', None])
    except: None
    try:
        for p in arch.policy_selector_evo_task_pipes: p[0].send(['TERMINATE', None])
    except: None
    print('Finished Running ', args.savetag)
    exit(0)











