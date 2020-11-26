import numpy as np, sys


class RoverDomainPython:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args

		from envs.rover_domain.rover_domain_python import RoverDomainVel

		self.universe = [] #Universe - collection of all envs running in parallel
		for _ in range(num_envs):
			env = RoverDomainVel(args.config)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:])
			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(info)

		joint_obs = np.stack(joint_obs, axis=1)


		try:
			joint_reward = np.stack(joint_reward, axis=1)
		except:
				import pdb
				pdb.set_trace()

		return joint_obs, joint_reward, joint_done, joint_global

	def render(self):

		rand_univ = np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].render()
		except: 'Error rendering'



class RoverHeterogeneousSequential:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs, reward_type_idx, type):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.reward_type_idx = reward_type_idx

		#from envs.rover_domain.rover_heterogeneous_sequential import RoverDomainHeterogeneousSequential
		from envs.rover_domain.rover_domain_sequential import SequentialPOIRD

		self.universe = [] #Universe - collection of all envs running in parallel
		for i in range(num_envs):
			env = RoverDomainHeterogeneousSequential(args.config, reward_type_idx, type, i)
			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action, action_choice): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			next_state, reward, done, info = env.step(action[:,universe_id,:], action_choice[:, universe_id])

			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(info)

			#if(len(next_state[0])!=111 or len(next_state[1])!=111 or len(next_state[2])!=111 or len(next_state[3])!=111):
			#	print("NO...................", len(next_state[0]), len(next_state[1]), len(next_state[2]), len(next_state[3]))
				#next_state, reward, done, info = env.step(action[:, universe_id, :])

		try: joint_obs = np.stack(joint_obs, axis=1)
		except:
			print("Not happening", (len(joint_obs), len(joint_obs[0])))

		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):
		if self.reward_type_idx == -1: # for overall test case
			print("Rendering System visualization:")
		else: # for primitive test
			print("Rendering:", self.args.config.reward_types[self.reward_type_idx])
		rand_univ = 0 #np.random.randint(0, len(self.universe))
		self.universe[rand_univ].render()



	def viz(self):
		if self.reward_type_idx == -1: # for overall test case
			print("Visualizing System visualization:")
		else: # for primitive test
			print("Visualizing:", self.args.config.reward_types[self.reward_type_idx])

		rand_univ = 0#np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].viz(save=False, fname='path')
		except: 'Error rendering'



########### This is python implemented wrapper of Connor's work

class SequentialPOIRD:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, args, num_envs, reward_type_idx, type):
		"""
		A base template for all environment wrappers.
		"""
		#Initialize world with requiste params
		self.args = args
		self.reward_type_idx = reward_type_idx

		#from envs.rover_domain.rover_heterogeneous_sequential import RoverDomainHeterogeneousSequential
		from envs.rover_domain.rover_domain_sequential import SequentialPOIRoverDomain

		self.universe = [] #Universe - collection of all envs running in parallel
		for i in range(num_envs):
			#env = SequentialPOIRoverDomain(6, 4, 50, 50, [0, 0, 1, 2], {0: None, 1: [0], 2: None, 3: [2]}, len(self.args.config.reward_types))
			#env = SequentialPOIRoverDomain(2, 4, 50, 50, [0, 1, 2, 3], {0: None, 1: [0], 2: [1], 3: [2]}, len(self.args.config.reward_types))
			# num_rovers
			# num_poi
			# num_steps
			# setup_size
			# poi_types
			# poi_sequence
			# num_reward_types
			env = SequentialPOIRoverDomain(self.args.config.num_uavs,  # num_rovers
										   self.args.config.num_poi,  # num_poi
										   self.args.config.ep_len,  # num_steps
										   self.args.config.dim_x,  # setup_size
										   self.args.config.poi_types,  # poi_types
										   self.args.config.poi_sequence,  # poi_sequence
										   len(self.args.config.reward_types),  # num_reward_types
										   self.args.config.coupling_uav, # coupling factor
										   i)  # for random seed

			self.universe.append(env)

		#Action Space
		self.action_low = -1.0
		self.action_high = 1.0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		joint_obs = []
		for env in self.universe:
			obs = env.reset()
			obs = obs.reshape(obs.shape[0], -1)
			joint_obs.append(obs)

		joint_obs = np.stack(joint_obs, axis=1)
		#returns [agent_id, universe_id, obs]

		return joint_obs


	def step(self, action, action_choice): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		joint_obs = []; joint_reward = []; joint_done = []; joint_global = []
		for universe_id, env in enumerate(self.universe):
			#next_state, reward, done, info = env.step(action[:,universe_id,:], action_choice[:, universe_id])
			next_state, reward, done, global_reward = env.step(action[:,universe_id,:])

			next_state = next_state.reshape(next_state.shape[0], -1)

			joint_obs.append(next_state); joint_reward.append(reward); joint_done.append(done); joint_global.append(global_reward)

			#if(len(next_state[0])!=111 or len(next_state[1])!=111 or len(next_state[2])!=111 or len(next_state[3])!=111):
			#	print("NO...................", len(next_state[0]), len(next_state[1]), len(next_state[2]), len(next_state[3]))
				#next_state, reward, done, info = env.step(action[:, universe_id, :])

		try: joint_obs = np.stack(joint_obs, axis=1)
		except:
			print("Not happening", (len(joint_obs), len(joint_obs[0])))

		joint_reward = np.stack(joint_reward, axis=1)

		return joint_obs, joint_reward, joint_done, joint_global



	def render(self):
		if self.reward_type_idx == -1: # for overall test case
			print("Rendering System visualization:")
		else: # for primitive test
			print("Rendering:", self.args.config.reward_types[self.reward_type_idx])
		rand_univ = np.random.randint(0, len(self.universe))
		self.universe[rand_univ].render()



	def viz(self):
		if self.reward_type_idx == -1: # for overall test case
			print("Visualizing System visualization:")
		else: # for primitive test
			print("Visualizing:", self.args.config.reward_types[self.reward_type_idx])

		rand_univ = np.random.randint(0, len(self.universe))
		try: self.universe[rand_univ].viz(save=False, fname='path')
		except: 'Error rendering'



