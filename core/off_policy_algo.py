import torch, os, random
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from core import mod_utils as utils
from core.models import Actor, QNetwork
import pdb

class TD3(object):
	"""Classes implementing TD3 and DDPG off-policy learners

		 Parameters:
			   args (object): Parameter class


	 """
	def __init__(self, algo_name, state_dim, action_dim, hidden_size, actor_lr, critic_lr, gamma, tau, savetag, foldername, actualize, use_gpu, init_w = True):

		self.algo_name = algo_name; self.gamma = gamma; self.tau = tau; self.total_update = 0; self.actualize = actualize; self.use_gpu = use_gpu
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'policy_loss_'+savetag, 'alz_score'+savetag,'alz_policy'+savetag], '.csv', save_iteration=1000, conv_size=1000)

		#Initialize actors
		self.policy = Actor(state_dim, action_dim, hidden_size, policy_type='DeterministicPolicy')
		if init_w: self.policy.apply(utils.init_weights)
		self.policy_target = Actor(state_dim, action_dim, hidden_size, policy_type='DeterministicPolicy')
		utils.hard_update(self.policy_target, self.policy)
		self.policy_optim = Adam(self.policy.parameters(), actor_lr)


		self.critic = QNetwork(state_dim, action_dim,hidden_size)
		if init_w: self.critic.apply(utils.init_weights)
		self.critic_target = QNetwork(state_dim, action_dim, hidden_size)
		utils.hard_update(self.critic_target, self.critic)
		self.critic_optim = Adam(self.critic.parameters(), critic_lr)

		'''
		if actualize:
			self.ANetwork = ActualizationNetwork(state_dim, action_dim, hidden_size)
			if init_w: self.ANetwork.apply(utils.init_weights)
			self.actualize_optim = Adam(self.ANetwork.parameters(), critic_lr)
			self.actualize_lr = 0.2
			if use_gpu: self.ANetwork.cuda()
		'''
		self.loss = nn.MSELoss()

		if use_gpu:
			self.policy_target.cuda(); self.critic_target.cuda(); self.policy.cuda(); self.critic.cuda()
		self.num_critic_updates = 0

		#Statistics Tracker
		#self.action_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.alz_score = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.alz_policy = {'min':None, 'max': None, 'mean':None, 'std':None}
		#self.val = {'min':None, 'max': None, 'mean':None, 'std':None}
		#self.value_loss = {'min':None, 'max': None, 'mean':None, 'std':None}


	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, global_reward, num_epoch=1, **kwargs):
		"""Runs a step of Bellman upodate and policy gradient using a batch of experiences

			 Parameters:
				  state_batch (tensor): Current States
				  next_state_batch (tensor): Next States
				  action_batch (tensor): Actions
				  reward_batch (tensor): Rewards
				  done_batch (tensor): Done batch
				  num_epoch (int): Number of learning iteration to run with the same data

			 Returns:
				   None

		 """

		self.gamma = 0.5 # todo: just to check it

		if isinstance(state_batch, list):
			state_batch = torch.cat(state_batch);
			next_state_batch = torch.cat(next_state_batch);
			action_batch = torch.cat(action_batch);
			reward_batch = torch.cat(reward_batch);
			done_batch = torch.cat(done_batch);
			global_reward = torch.cat(global_reward)

		for _ in range(num_epoch):
			########### CRITIC UPDATE ####################

			#Compute next q-val, next_v and target
			with torch.no_grad():
				#Policy Noise
				#pdb.set_trace()

				policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
				policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

				#Compute next action_batch
				next_action_batch = self.policy_target.clean_action(next_state_batch, return_only_action=True) + policy_noise.cuda() if self.use_gpu else policy_noise
				next_action_batch = torch.clamp(next_action_batch, -1, 1)

				#Compute Q-val and value of next state masking by done
				q1, q2 = self.critic_target.forward(next_state_batch, next_action_batch)

				q1 = (1 - done_batch) * q1
				q2 = (1 - done_batch) * q2
				#next_val = (1 - done_batch) * next_val

				#Select which q to use as next-q (depends on algo)
				if self.algo_name == 'TD3' or self.algo_name == 'TD3_actor_min': next_q = torch.min(q1, q2)
				elif self.algo_name == 'DDPG': next_q = q1
				elif self.algo_name == 'TD3_max': next_q = torch.max(q1, q2)

				#Compute target q and target val
				target_q = reward_batch + (self.gamma * next_q)
				#if self.args.use_advantage: target_val = reward_batch + (self.gamma * next_val)

			self.critic_optim.zero_grad()
			current_q1, current_q2 = self.critic.forward((state_batch), (action_batch))
			utils.compute_stats(current_q1, self.q)

			dt = self.loss(current_q1, target_q)
			# if self.args.use_advantage:
			#     dt = dt + self.loss(current_val, target_val)
			#     utils.compute_stats(current_val, self.val)

			if self.algo_name == 'TD3' or self.algo_name == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
			utils.compute_stats(dt, self.q_loss)

			# if self.args.critic_constraint:
			#     if dt.item() > self.args.critic_constraint_w:
			#         dt = dt * (abs(self.args.critic_constraint_w / dt.item()))
			dt.backward()

			self.critic_optim.step()
			self.num_critic_updates += 1
			#print("TD3 updated")

			#Delayed Actor Update
			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0:

				actor_actions = self.policy.clean_action(state_batch, return_only_action=False)

				# # Trust Region constraint
				# if self.args.trust_region_actor:
				#     with torch.no_grad(): old_actor_actions = self.actor_target.forward(state_batch)
				#     actor_actions = action_batch - old_actor_actions


				Q1, Q2 = self.critic.forward(state_batch, actor_actions)

				# if self.args.use_advantage: policy_loss = -(Q1 - val)
				policy_loss = -Q1

				utils.compute_stats(policy_loss,self.policy_loss)
				policy_loss = policy_loss.mean()

				if policy_loss.item() > 1000:
					print( policy_loss.item(), Q1, Q2)
					#pdb.set_trace()


				self.policy_optim.zero_grad()
				policy_loss.backward(retain_graph=True)
				#nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
				# if self.args.action_loss:
				#     action_loss = torch.abs(actor_actions-0.5)
				#     utils.compute_stats(action_loss, self.action_loss)
				#     action_loss = action_loss.mean() * self.args.action_loss_w
				#     action_loss.backward()
				#     #if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
				self.policy_optim.step()


			# if self.args.hard_update:
			#     if self.num_critic_updates % self.args.hard_update_freq == 0:
			#         if self.num_critic_updates % self.args.policy_ups_freq == 0: self.hard_update(self.actor_target, self.actor)
			#         self.hard_update(self.critic_target, self.critic)


			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0: utils.soft_update(self.policy_target, self.policy, self.tau)
			utils.soft_update(self.critic_target, self.critic, self.tau)

			self.total_update += 1
			#if self.agent_id == 0:
		    #   self.tracker.update([self.q['mean'], self.q_loss['mean'], self.policy_loss['mean'],self.alz_score['mean'], self.alz_policy['mean']] ,self.total_update)
