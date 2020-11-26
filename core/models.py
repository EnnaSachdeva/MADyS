import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from core import mod_utils as utils
import random
import pdb


################################################################################
################  DQN model for Policy selector ###################
################################################################################

class DQN_model(nn.Module):
	def __init__(self, input_dims, hidden_size, output_dims, reward_types, num_agents): # here the ouput dimension is centralized policy which gives the policy choice for each agent
		super(DQN_model, self).__init__()

		self.num_agents = num_agents
		self.reward_types = reward_types
		self.layernorm1 = nn.LayerNorm(input_dims)
		self.linear1 = nn.Linear(input_dims, hidden_size)
		self.layernorm2 = nn.LayerNorm(hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.layernorm3 = nn.LayerNorm(hidden_size)
		self.linear3 = nn.Linear(hidden_size, output_dims) # output: number of policy choices

		self.apply(weights_init_policy_fn)


	def forward(self, x):
		'''
		x = self.linear1(x)
		x = torch.tanh(x)
		x = self.layernorm2(x)
		x = self.linear2(x)
		x = torch.tanh(x)
		x = self.layernorm3(x)
		x = self.linear3(x)
		x = torch.tanh(x)
		'''

		'''
		x = self.linear1(x)
		x = torch.tanh(x)
		x = self.linear2(x)
		x = torch.tanh(x)
		x = self.linear3(x)
		x = torch.tanh(x)
		'''
		x = self.linear1(x)
		x = F.relu(x)
		x = self.linear2(x)
		x = F.relu(x)
		x = self.linear3(x)

		return x


	def policy_selector(self, state, epsilon): # here, we get for each agent individually
		#epsilon = 0.0
		all_policies = self.forward(state).detach().numpy()
		all_policies = utils.to_tensor(np.array(all_policies))

		policy_choice = []

		#if epsilon > 0:
		#	print(epsilon)

		if (random.random() < epsilon):
			#temp_policy_choice = torch.zeros([all_policies.size()[0], all_policies.size()[1]])
			policy_choice = torch.randint(0, all_policies.size()[1], (all_policies.size()[0], ))

			'''
			for k in range(self.num_agents):
				temp = k*len(self.reward_types)
				policy_choice.append(np.random.choice(all_policies[temp: temp+3])) # random
			'''

		else:
			policy_choice = torch.argmax(all_policies, dim = 1) # finding the policy with the max q value
			'''
			for k in range(self.num_agents): # no exploration
				_, idx = torch.max(all_policies[k, :, :], dim = 1) # todo: start from here
				policy_choice.append(idx)
				#policy_choice.append(torch.max(all_policies[k, :, :], dim=1))
			'''
		return policy_choice.float()


	def noisy_action(self, state, return_only_action= True):

		if self.policy_type == 'GaussianPolicy':
			mean, log_std = self.clean_action(state, return_only_action= False)
			std = log_std.exp()
			normal = Normal(mean, std)
			x_t = normal.rsample()
			action = torch.tanh(x_t)

			if return_only_action: return action
			log_prob = normal.log_prob(x_t)
			log_prob -= torch.log(1 - action.pow(2) + epsilon)
			log_prob = log_prob.sum(-1, keepdim = True)

			return action, log_prob, x_t, mean, log_std


################################################################################
################ Actor model for Multiagent Policies ###################
################################################################################

LOG_SIG_MAX = 5
LOG_SIG_MIN = -10
epsilon = 1e-6


class Actor(nn.Module):

	def __init__(self, num_inputs, num_actions, hidden_size, policy_type):
		super(Actor, self).__init__()
		self.policy_type = policy_type

		if self.policy_type == "GaussianPolicy":
			self.linear1 = nn.Linear(num_inputs, hidden_size)
			self.linear2 = nn.Linear(hidden_size, hidden_size) #todo

			self.mean_linear = nn.Linear(hidden_size, num_actions)
			self.log_std_linear = nn.Linear(hidden_size, num_actions)

			self.apply(weights_init_policy_fn)


		elif self.policy_type == 'DeterministicPolicy':
			self.linear1 = nn.Linear(num_inputs, hidden_size)
			self.linear2 = nn.Linear(hidden_size, hidden_size)
			self.mean = nn.Linear(hidden_size, num_actions)
			self.noise = torch.Tensor(num_actions)
			self.apply(weights_init_policy_fn)


	def clean_action(self, state, return_only_action=True):

		if len(list(state.size())) < 2:
			state = state.unsqueeze(0)

		if self.policy_type == 'GaussianPolicy':
			x = torch.tanh(self.linear1(state))
			x = torch.tanh(self.linear2(x))
			mean = self.mean_linear(x)
			if return_only_action:
				return torch.tanh(mean)


			log_std = self.log_std_linear(x)
			log_std = torch.clamp(log_std, min = LOG_SIG_MIN, max= LOG_SIG_MAX)
			return mean, log_std

		elif self.policy_type == 'DeterministicPolicy':
			x = torch.tanh(self.linear1(state))
			x = torch.tanh(self.linear2(x))
			mean = torch.tanh(self.mean(x))

			return mean

	def noisy_action(self, state, return_only_action= True):

		if self.policy_type == 'GaussianPolicy':
			mean, log_std = self.clean_action(state, return_only_action= False)
			std = log_std.exp()
			normal = Normal(mean, std)
			x_t = normal.rsample()
			action = torch.tanh(x_t)

			if return_only_action: return action
			log_prob = normal.log_prob(x_t)
			log_prob -= torch.log(1 - action.pow(2) + epsilon)
			log_prob = log_prob.sum(-1, keepdim = True)

			return action, log_prob, x_t, mean, log_std


		elif self.policy_type == 'DeterministicPolicy':
			mean = self.clean_action(state)
			action = mean + self.noise.normal_(0., std = 0.4)

			if return_only_action: return action
			else: return action, torch.tensor(0.), torch.tensor(0.), mean, torch.tensor(0.)


################################################################################
################ Critic model for Multiagent Policies ###################
################################################################################

class QNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size):
		super(QNetwork, self).__init__()

		#Q1 architecture
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		# Q2 architecture
		self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear5 = nn.Linear(hidden_size, hidden_size)
		self.linear6 = nn.Linear(hidden_size, 1)

		self.apply(weights_init_value_fn)


	def forward(self, state, action):
		x1 = torch.cat([state, action], 1)
		x1 = torch.tanh(self.linear1(x1))
		x1 = torch.tanh(self.linear2(x1))
		#print("This is done")
		x1 = self.linear3(x1)

		x2 = torch.cat([state, action], 1)
		x2 = torch.tanh(self.linear4(x2))
		x2 = torch.tanh(self.linear5(x2))
		#print("This is done")
		x2 = self.linear6(x2)

		#x2 = torch.cat([state, action], 1)
		#x2 = torch.tanh(self.linear4(x2))
		#x2 = torch.tanh(self.linear5(x2))
		#x2 = self.linear6(x2)


		return x1, x2


################################################################################
################### Initialize weights of the models #####################
################################################################################

def weights_init_policy_fn(m):
	classname = m.__class__.__name__

	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain = 0.5)
		torch.nn.init.constant_(m.bias, 0)



def weights_init_value_fn(m):
	classname = m.__class__.__name__

	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain = 1)
		torch.nn.init.constant_(m.bias, 0)