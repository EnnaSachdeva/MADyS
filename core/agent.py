from core.off_policy_algo import TD3
from torch.multiprocessing import Manager
from core.neuroevolution import SSNE
from core.models import DQN_model, Actor, QNetwork
from core.policy_selector_algo import DQN
import core.mod_utils as mod
from core.buffer import Buffer

import numpy as np
import random, sys
import torch

import pdb

################################################################################
################  ERL agent of Policy Selectors ###################
################################################################################

class PolicySelectorAgent:

	def __init__(self, args, id):
		self.args = args

		self.evolver = SSNE(self.args)

		#### Initialize Population
		self.manager = Manager()
		self.popn = self.manager.list()

		self.policy_selector_action_dim = self.args.policy_selector_action_dim + self.args.cardinal_actions # including cardinal actions as well
		## Initialize the Population of networks
		for _ in range(self.args.policy_selector_popn_size):
			self.popn.append(DQN_model(self.args.state_dim, self.args.hidden_size, self.policy_selector_action_dim,
									   self.args.config.reward_types, self.args.config.num_agents))
			self.popn[-1].eval()  # todo: why is this added????

		## Initialize Policy Gradient Algo
		self.algo = DQN(args)

		## Rollout actor
		self.rollout_actor = self.manager.list()
		self.rollout_actor.append(
			DQN_model(self.args.state_dim, self.args.hidden_size, self.policy_selector_action_dim,
					  self.args.config.reward_types,
					  self.args.config.num_agents))  # todo, this rollout will be a bit different

		# Initialize buffer
		# self.buffer = Buffer(args.DQN_buffer_capacity, buffer_gpu=False, buffer_type = 'Policy_selector', filter_c=args.filter_c)  # this is for one agent

		# Agent metrics
		self.fitness = [[] for _ in range(args.policy_selector_popn_size)]

		# Best Policy hall of fame
		self.champ_ind = 0

		self.updates = 0

	def update_parameters(self, buffer):

		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': -1.0,
				   'action_high': 1.0, 'use_dpp': False}

		buffer.refresh() # todo: not needed twice

		if buffer.__len__() < 10 * self.args.DQN_batch_size:
			return

		buffer.tensorify() #todo: not needed twice

		for _ in range(int(
				self.args.gradperstep * buffer.pg_frames_policy_selector)):  # how many times of the new frames added, should the update happen
			# print("Policy selector agent")

			state, next_state, _, policy_choice, _, done, global_reward = buffer.sample(self.args.DQN_batch_size,
																						pr_rew=self.args.priority_rate,
																						pr_global=self.args.priority_rate,
																						type='policy_selector')

			# scale the global reward
			global_reward *= self.args.reward_scaling

			self.updates += 1

			if self.args.use_gpu:
				state = state.cuda();
				next_state = next_state.cuda();
				# action = action.cuda();
				policy_choice = policy_choice.cuda();
				done = done.cuda();
				global_reward = global_reward.cuda();

			# pdb.set_trace()
			self.algo.update_parameters(state, next_state, policy_choice, done, global_reward, self.updates)

		buffer.pg_frames_policy_selector = 0

	#### Evolving the weights of the ERL Population part
	def evolve(self):

		if self.args.policy_selector_popn_size > 1:
			states = None

			# alloting indices to each network in the population
			net_inds = [i for i in range(len(self.popn))]

			# Evolve:
			if self.args.policy_selector_rollout_size > 0:
				self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitness, [self.rollout_actor[0]], states)

			else:
				self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitness, [], states)

		self.fitness = [[] for _ in range(self.args.policy_selector_popn_size)]

	def update_rollout_actor(self):  # todo: update DQN actors
		for actor in self.rollout_actor:
			self.algo.policy.cpu()
			mod.hard_update(actor, self.algo.policy)

			if self.args.use_gpu:
				self.algo.policy.cuda()


################################################################################
###################  Multiagent Local primitive learners #######################
################################################################################


class PrimitiveAgent:
	def __init__(self, args, id):
		self.args = args
		self.id = id

		self.algo = TD3(args.algo_name, args.state_dim, args.agent_action_dim, args.hidden_size, args.actor_lr,
						args.critic_lr, args.gamma, args.tau, args.savetag, args.policy_selector_q_save, args.actualize,
						args.use_gpu,
						args.init_w)

		self.manager = Manager()

		self.rollout_actor = self.manager.list()

		if self.args.config.low_pretrained: # if pre-trained agents are being used
			#filename = self.args.save_foldername + "primitive_models/" + str(self.args.config.reward_types[id]) + "_"+self.args.actor_fname+"primitive"
			filename = "primitive_models/" + str(self.args.config.reward_types[id]) + "_actor_pop0_roll0_env-rover_heterogeneous_fire_truck_uav_action_same_seed1_rewardmultiple_uav4_trucks0_coupling-uav2_coupling-truck0_obs-uav800_obs-truck800primitive"
			m = torch.load(filename)
			temp_model = Actor(args.state_dim, args.agent_action_dim, args.hidden_size, policy_type='DeterministicPolicy')
			temp_model.load_state_dict(m)
			self.rollout_actor.append(temp_model)
		else:
			self.rollout_actor.append(
				Actor(args.state_dim, args.agent_action_dim, args.hidden_size, policy_type='DeterministicPolicy'))

	# self.buffer = Buffer(args.TD3_buffer_capacity, buffer_gpu = False, buffer_type = 'Primitive_agents', filter_c = args.filter_c) #this will be the common buffer, which will be used by the manager

	def update_parameters(self, buffer, reward_id):

		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 5, 'action_low': -1.0,
				   'action_high': 1.0, 'use_dpp': False}

		# primitive can sample batch from any of replay buffers

		'''
		random_num = np.random.randint(0, self.args.config.num_agents)
		buffer = whole_buffer[random_num]  # todo: merge all buffers together and then take the batch
		'''
		buffer.refresh()


		if buffer.__len__() < 10 * self.args.TD3_batch_size:
			return

		buffer.tensorify()
		for _ in range(int(self.args.gradperstep * buffer.pg_frames_primitives)):
			# print("Policy selector agent")
			'''
			random_num = np.random.randint(0, self.args.config.num_agents) # everytime sample from a new buffer
			buffer = whole_buffer[random_num]
			'''

			state, next_state, action, _, reward, done, global_reward = buffer.sample(self.args.TD3_batch_size,
																					  pr_rew=self.args.priority_rate,
																					  pr_global=self.args.priority_rate,
																					  type='primitives')

			# reward = reward[:, :, reward_id] # take reward corresponding to that id of reward
			if self.args.config.reward_scheme == 'multiple':
				reward = reward.squeeze(1)

			reward = reward[:, reward_id].unsqueeze(1)

			# reward = reward[:, reward_id].unsqueeze(1) # todo: does not work for tenrosify
			reward *= self.args.reward_scaling

			if self.args.use_gpu:
				state = state.cuda();
				next_state = next_state.cuda();
				action = action.cuda();
				reward = reward.cuda();
				done = done.cuda();
				global_reward = global_reward.cuda();

			self.algo.update_parameters(state, next_state, action, reward, done, global_reward, 1, **td3args)

		buffer.pg_frames_primitives = 0

	# copy the trained weights
	def update_rollout_actor(self):  # todo: update DQN actors
		for actor in self.rollout_actor:
			self.algo.policy.cpu()
			mod.hard_update(actor, self.algo.policy)

			if self.args.use_gpu:
				self.algo.policy.cuda()

################################################################################
############################## Test agent ######################################
################################################################################

class TestAgent:
	def __init__(self, args):
		self.args = args
		# self.id = id

		## Initialize Policy Gradient Algo
		self.policy_selector_algo = DQN(args)

		# todo: load the ERL champion model
		self.manager = Manager()

		self.rollout_actor = self.manager.list()
		self.policy_selector_action_dim = self.args.policy_selector_action_dim + self.args.cardinal_actions # including cardinal actions as well

		for agent_id in range(args.config.num_agents):
			self.rollout_actor.append(
				DQN_model(self.args.state_dim, self.args.hidden_size, self.policy_selector_action_dim,
						  self.args.config.reward_types, self.args.config.num_agents))

	# self.rollout_actor.append(Actor(args.state_dim, args.agent_action_dim, args.hidden_size, policy_type='DeterministicPolicy'))

	# self.buffer = Buffer(args.buffer_size, buffer_gpu = False, buffer_type = 'Primitive_agents', filter_c = args.filter_c) #this will be the common buffer, which will be used by the manager

	# copy champion weights to test model
	def get_champ(self, agents):
		for agent_id, agent in enumerate(agents):
			if self.args.policy_selector_popn_size <= 1:
				agent.update_rollout_actor()
				mod.hard_update(self.rollout_actor[agent_id], agent.rollout_actor[0])
			else:
				mod.hard_update(self.rollout_actor[agent_id], agent.popn[agent.champ_ind])
				#print("Agent champ index", agent_id, agent.champ_ind)# here the champion weight gets copied to the test agent
