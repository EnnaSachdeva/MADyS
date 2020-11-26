
import torch
import random, sys
from torch import nn
from torch.optim import Adam
from core.models import DQN_model
from core import mod_utils as utils
import pdb
from torch.nn.utils import clip_grad_norm_

class DQN:

	def __init__(self, args):

		init_w = True # todo: changed from True

		self.args= args

		### Statistics Tracker
		self.q_value = {'min': None, 'max': None, 'mean': None, 'std': None}
		self.target_q_value = {'min': None, 'max': None, 'mean': None, 'std': None}
		self.q_loss = {'min': None, 'max': None, 'mean': None, 'std': None}


		# todo: need to add policy losses
		self.tracker = utils.Tracker(self.args.savetag, ['policy_selector_q_'+self.args.savetag, 'policy_selector_qloss_'+self.args.savetag, 'policy_selector_policy_loss_'+self.args.savetag], '.csv', save_iteration=1000, conv_size=1000)

		self.policy = DQN_model(self.args.state_dim, self.args.hidden_size, len(self.args.config.reward_types) + self.args.cardinal_actions, self.args.config.reward_types, self.args.config.num_agents).cuda()
		self.policy_target = DQN_model(self.args.state_dim, self.args.hidden_size, len(self.args.config.reward_types) + self.args.cardinal_actions, self.args.config.reward_types, self.args.config.num_agents).cuda()

		self.policy_target.load_state_dict(self.policy.state_dict())
		self.policy_target.eval()

		if init_w:
			self.policy.apply(utils.init_weights)

		utils.hard_update(self.policy_target, self.policy)

		self.policy_optim = Adam(self.policy.parameters(), lr = self.args.learning_rate)
		self.loss = nn.MSELoss()

		if self.args.use_gpu:
			self.policy_target.cuda();
			self.policy.cuda();

	def act(self, state, epsilon = None):
		if (epsilon is None): epsilon = self.config.epsilon_min
		if random.random() > epsilon:
			state = torch.tensor(state, dtype = torch.float).unsqeeze(0)

			if self.config.use_cuda:
				state = state.cuda()

			q_value = self.model.forward(state)
			actions = q_value.max(1)[1].item()
		else:
			action = random.randrange(self.config.action_dim)
		return action


	def update_parameters(self, state_batch, next_state_batch, policy_choice_batch, done_batch, global_reward_batch, updates):

		'''
		if isinstance(state_batch, list):


			# todo: we don't need this
			state_batch = torch.cat(state_batch)
			next_state_batch = torch.cat(next_state_batch)
			#action_batch = torch.cat(action_batch)
			policy_choice_batch = torch.cat(policy_choice_batch)
			if self.args.use_done_mask: done_batch = torch.cat(done_batch)
			global_reward_batch = torch.cat(global_reward_batch)
		'''

		# Load everything to GPU
		if self.args.is_cuda:
			state_batch = state_batch.cuda();
			next_state_batch = next_state_batch.cuda();
			#action_batch = action_batch.cuda();
			policy_choice = global_reward_batch.cuda();
			if self.args.use_done_mask: done_batch = done_batch.cuda()

		policy_choice_batch = policy_choice_batch.long()

		# finding the q values of current state and next state
		state_q_values = self.policy.forward(state_batch).cuda()  # gives actions of each agent
		state_action_q_value = torch.gather(state_q_values, 1, policy_choice_batch)

		next_state_q_values = self.policy.forward(next_state_batch).cuda()
		next_state_target_q_values = self.policy_target.forward(next_state_batch).cuda()

		#next_state_action_q_value = torch.gather(next_state_target_q_values, 1, next_state_q_values.max(1)[1].unsqueeze(1))
		next_state_action_q_value = next_state_target_q_values.max(1)[0].unsqueeze(1)

		temp_done_batch = torch.zeros([done_batch.size()[0], done_batch.size()[1]])
		temp_global_reward_batch = torch.zeros([global_reward_batch.size()[0], global_reward_batch.size()[1]])

		if self.args.use_done_mask: next_state_action_q_value = next_state_action_q_value * (1 - done_batch)  # Done mask

		target_q_value = global_reward_batch + self.args.gamma * next_state_action_q_value  # todo: need to change this global_batch to local reward batch, if it does not seem to work

		loss = self.loss(state_action_q_value, target_q_value)  # for loss, always one the last 1 dimension should be removed, else it always creates problem, a*b*1 should be squeezed down to a*b

		batch_size = state_batch.size()[0]
		# loss = loss/batch_size

		utils.compute_stats(loss, self.q_loss)

		self.policy_optim.zero_grad()

		# loss.requires_grad = True

		loss.backward()

		self.policy_optim.step()

		# Get statistics
		utils.compute_stats(state_action_q_value, self.q_value)
		utils.compute_stats(target_q_value, self.target_q_value)
		#pdb.set_trace()
		#print("DQN updated")
		## Soft update

		if updates % self.args.dqn_target_update_interval == 0:
			utils.hard_update(self.policy_target, self.policy)

		utils.soft_update(self.policy_target, self.policy, self.args.tau)

		#if torch.max(loss).item() > 1000:
		#	print(loss, torch.min(loss).item(), torch.max(loss).item(), torch.mean(loss).item(), torch.std(loss).item())
		#	pdb.set_trace()


		'''
		###### todo: need to see this, Global reward needs to be added
		# Load everything to GPU
		if self.args.is_cuda:
			state_batch = state_batch.cuda();
			next_state_batch = next_state_batch.cuda();
			#action_batch = action_batch.cuda();
			policy_choice = global_reward_batch.cuda();
			if self.args.use_done_mask: done_batch = done_batch.cuda()
		'''

		'''
		# finding the q values of current state and next state
		q_values = self.policy.forward(state_batch).cuda() # gives actions of each agent
		next_q_values = self.policy.forward(next_state_batch).cuda()
		next_q_state_values = self.policy_target.forward(next_state_batch).cuda()


		policy_choice_batch = policy_choice_batch.long()

		q_value = torch.gather(q_values, 1, policy_choice_batch)

		next_q_value = torch.gather(next_q_state_values, 1, next_q_values.max(1)[1].unsqueeze(1))

		temp_done_batch = torch.zeros([done_batch.size()[0], done_batch.size()[1]])
		temp_global_reward_batch = torch.zeros([global_reward_batch.size()[0], global_reward_batch.size()[1]])


		if self.args.use_done_mask: next_q_value = next_q_value * (1 - done_batch)  # Done mask

		target_q_value = global_reward_batch + self.args.gamma * next_q_value # todo: need to change this global_batch to local reward batch, if it does not seem to work

		loss = self.loss(q_value, target_q_value) # for loss, always one the last 1 dimension should be removed, else it always creates problem, a*b*1 should be squeezed down to a*b

		batch_size = state_batch.size()[0]
		#loss = loss/batch_size

		utils.compute_stats(loss, self.q_loss)

		self.policy_optim.zero_grad()


		#loss.requires_grad = True

		loss.backward()

		self.policy_optim.step()
		
		
		# Get statistics
		utils.compute_stats(q_value, self.q_value)
		utils.compute_stats(target_q_value, self.target_q_value)
		print("DQN updated")

		if torch.max(loss).item() > 1000:
			print(loss, torch.min(loss).item(), torch.max(loss).item(), torch.mean(loss).item(), torch.std(loss).item())
			pdb.set_trace()

		'''
		'''
		state_action_values = self.policy.forward(state_batch).cuda()
		next_state_values = self.policy_target.forward(next_state_batch).cuda()

		expected_state_action_values = global_reward_batch + self.args.gamma * next_state_values
		loss = self.loss(state_action_values, expected_state_action_values)

		utils.compute_stats(loss, self.q_loss)

		self.policy_optim.zero_grad()
		loss.backward()
		self.policy_optim.step()

		# Get statistics
		utils.compute_stats(state_action_values, self.q_value)
		utils.compute_stats(expected_state_action_values, self.target_q_value)
		'''
		#############################
		'''
		if self.policy_loss['mean] ==float('nan'):
			print("Gradients of layers of DQN############")
			print(self.policy.linear1.weight.grad)
			print(self.policy.linear2.weight.grad)
			print(self.policy.linear3.weight.grad)
			print("Q value", self.q_value)
			print("Target Q value", self.target_q_value)
			print("Loss", self.policy_loss)

			pdb.set_trace()
		'''

		'''
		if updates % self.args.dqn_target_update_interval == 0:
			utils.hard_update(self.policy_target, self.policy)
		'''
#return loss.item()



