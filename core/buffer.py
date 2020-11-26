import numpy as np
import random
import torch
from torch.multiprocessing import Manager
from core.mod_utils import compute_stats
import pdb

class Buffer():


	def __init__(self, capacity, buffer_gpu, buffer_type, filter_c = None):

		'''

		:param capacity:
		:param buffer_gpu:
		:param buffer_type: Primitive_agents, and Policy selector
		:param filter_c:

		Policy selector buffer: <joint state, joint_next_state, joint action, primitive policy index, done, global reward>

		Primitive policies buffer: <agent's state, agent's next state, agent's action, agent's local reward, done, global reward>

		'''

		self.capacity = capacity
		self.buffer_gpu = buffer_gpu
		self.buffer_type = buffer_type

		self.manager = Manager()
		#self.tuples = self.manager.list()

		self.tuples = self.manager.list()
		self.s = []
		self.ns = []
		self.a = []
		self.policy_choice = []
		self.r = []
		self.done = []
		self.global_reward = []
		'''

		# for optimized buffer part
		self.tuples = self.manager.dict()

		self.tuples['s'] = None
		self.tuples['ns'] = None
		self.tuples['a'] = None
		self.tuples['pc'] = None
		self.tuples['r'] = None
		self.tuples['d'] = None
		self.tuples['gr'] = None
		'''
		'''
		self.tuples['s'] = self.manager.list()
		self.tuples['ns'] = self.manager.list()
		self.tuples['a'] = self.manager.list()
		self.tuples['pc'] = self.manager.list()
		self.tuples['r'] = self.manager.list()
		self.tuples['d'] = self.manager.list()
		self.tuples['gr'] = self.manager.list()
		'''
		'''
		self.s = torch.Tensor()
		self.ns = torch.Tensor()
		self.a = torch.Tensor()
		self.policy_choice = torch.Tensor()
		self.r = torch.Tensor()
		self.done = torch.Tensor()
		self.global_reward = torch.Tensor()
		'''
		#if self.buffer_type == 'Policy_selector':
		#	self.policy_choice = [] # adding policy choice reward only for policy selector buffer

		#if self.buffer_type == 'Primitive_agents':
		#	self.r = [] # adding local reward only for Primitive_agents

		# Temporary tensors that can be loaded in PU for fast sampling during gradient updates (updated each gen) --> Faster sampling
		self.sT= None
		self.nsT = None
		self.aT = None
		self.policy_choiceT = None
		self.rT = None
		self.doneT = None
		self.global_rewardT = None

		#if self.buffer_type == 'Policy_selector':
		#	self.policy_choiceT = None

		#if self.buffer_type ==  'Primitive_agents':
		#	self.rT = None

		self.pg_frames_policy_selector = 0
		self.pg_frames_primitives = 0

		self.total_frames = 0

		#Priority indices
		self.top_r = None
		self.top_g = None

		# Stats
		self.rstats = {'min': None, 'max': None, 'mean': None, 'std': None}
		self.gstats = {'min': None, 'max': None, 'mean': None, 'std': None}


	def data_filter (self, exp):

		self.s.append(exp[0])
		self.ns.append(exp[1])
		self.a.append(exp[2])
		self.policy_choice.append(exp[3])
		self.r.append(exp[4])
		self.done.append(exp[5])
		self.global_reward.append(exp[6])

		#if self.buffer_type == 'Primitive_agents':
		#	self.r.append(exp[3]) # local reward for primitive policies
		#else:
		#	self.policy_choice.append(exp[3]) # Policy index for primitive policies

		self.pg_frames_primitives +=1
		self.pg_frames_policy_selector +=1
		self.total_frames +=1


	def concatenate_tensor(self, exp):

		torch.cat(self.s, exp[0])
		torch.cat(self.ns, exp[1])


		self.s.append()
		self.ns.append(exp[1])
		self.a.append(exp[2])
		self.policy_choice.append(exp[3])
		self.r.append(exp[4])
		self.done.append(exp[5])
		self.global_reward.append(exp[6])

		#if self.buffer_type == 'Primitive_agents':
		#	self.r.append(exp[3]) # local reward for primitive policies
		#else:
		#	self.policy_choice.append(exp[3]) # Policy index for primitive policies

		self.pg_frames +=1
		self.total_frames +=1


	def add(self, tensor_list):
		count = 0
		#pdb.set_trace()
		self.s = torch.cat((self.s, torch.stack(tensor_list['s']).squeeze(1)), 0)
		self.ns = torch.cat((self.ns, torch.stack(tensor_list['ns']).squeeze(1)), 0)
		self.a = torch.cat((self.a, torch.stack(tensor_list['a']).squeeze(1)), 0)
		self.policy_choice = torch.cat((self.policy_choice, torch.stack(tensor_list['pc']).squeeze(1)), 0)
		self.r = torch.cat((self.r, torch.stack(tensor_list['r']).squeeze(1)), 0)
		self.done = torch.cat((self.done, torch.stack(tensor_list['d']).squeeze(1)), 0)
		self.global_reward = torch.cat((self.global_reward, torch.stack(tensor_list['gr']).squeeze(1)), 0)
		#print("####### Total time taken to concatenate ", count, "tensors is ", time.time() - start)

		self.pg_frames += len(tensor_list['s'])
		self.total_frames += len(tensor_list['s'])

		#print("BUFFER LENGTH", self.total_frames, len(self.s))
		#pdb.set_trace()

		'''
		temp_s = []; temp_ns = []; temp_a = []; temp_pc = []; temp_r = []; temp_d = []; temp_gr=[]
		for tensor in tensor_list:
			temp_s.append(tensor[0])
			temp_ns.append(tensor[1])
			temp_a.append(tensor[2])
			temp_pc.append(tensor[3])
			temp_r.append(tensor[4])
			temp_d.append(tensor[5])
			temp_gr.append(tensor[6])

			self.pg_frames +=1
			self.total_frames +=1
			count += 1

		self.s = torch.cat((self.s, torch.stack(temp_s)), 0)
		self.ns = torch.cat((self.ns, torch.stack(temp_ns)), 0)
		self.a = torch.cat((self.a, torch.stack(temp_a)), 0)
		self.policy_choice = torch.cat((self.policy_choice, torch.stack(temp_pc)), 0)
		self.r = torch.cat((self.r, torch.stack(temp_r)), 0)
		self.done = torch.cat((self.done, torch.stack(temp_d)), 0)
		self.global_reward = torch.cat((self.global_reward, torch.stack(temp_gr)), 0)

		print("####### Total time taken to concatenate ", count, "tensors is ", time.time() - start)
		print("####### Time per tensor ", (time.time() - start) / count)
		pdb.set_trace()
		'''

		'''
		for tensors in tensor_list:
			self.s = torch.cat((self.s, tensors[0]), 0)
			self.ns = torch.cat((self.ns, tensors[1]), 0)
			self.a = torch.cat((self.a, tensors[2]), 0)
			self.policy_choice = torch.cat((self.policy_choice, tensors[3]), 0)

			self.r = torch.cat((self.r, tensors[4]), 0)
			self.done = torch.cat((self.done, tensors[5]), 0)
			self.global_reward = torch.cat((self.global_reward, tensors[6]), 0)

			self.pg_frames +=1
			self.total_frames +=1
			count +=1

		print("####### Total time taken to concatenate ", count, "tensors is ", time.time()-start)
		print("####### Time per tensor ", (time.time()-start)/count)
		'''


		#self.data_filter(exp)
	def refresh(self):
		'''
		for _ in range(len(self.tuples)):
			exp = self.tuples.pop()
			self.data_filter(exp)

		# Trim to make the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0);
			self.ns.pop(0);
			self.a.pop(0);
			self.policy_choice.pop(0);
			self.r.pop(0);
			self.done.pop(0);
			self.global_reward.pop(0)
		'''


		for _ in range(len(self.tuples)):
		#for _ in range(len(m)):
			#exp = self.tuples.pop()
			try:  exp = self.tuples.pop() #exp = self.tuples.pop()
			except: break
			self.data_filter(exp)


		# Trim to tmake the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0)
			self.ns.pop(0)
			self.a.pop(0)
			self.policy_choice.pop(0)
			self.r.pop(0)
			self.done.pop(0)
			self.global_reward.pop(0)

			#if self.buffer_type == 'Primitive_agents':
			#	self.r.pop(0)
			#else:
			#	self.policy_choice.pop(0)
		'''
		try:
			if (self.tuples['s'] != None and len(self.tuples['s']) > 0):
				self.s = torch.cat((self.s, torch.stack(self.tuples['s']).squeeze(1)), 0)
				self.ns = torch.cat((self.ns, torch.stack(self.tuples['ns']).squeeze(1)), 0)
				self.a = torch.cat((self.a, torch.stack(self.tuples['a']).squeeze(1)), 0)
				self.policy_choice = torch.cat((self.policy_choice, torch.stack(self.tuples['pc']).squeeze(1)), 0)
				self.r = torch.cat((self.r, torch.stack(self.tuples['r']).squeeze(1)), 0)
				self.done = torch.cat((self.done, torch.stack(self.tuples['d']).squeeze(1)), 0)
				self.global_reward = torch.cat((self.global_reward, torch.stack(self.tuples['gr']).squeeze(1)), 0)

				self.pg_frames_policy_selector += len(self.tuples['s'])
				self.pg_frames_primitives += len(self.tuples['s'])

				self.total_frames += len(self.tuples['s'])

				#if not(len(self.tuples['s']) == len(self.tuples['ns']) == len(self.tuples['a']) == len(self.tuples['pc']) == len(self.tuples['r']) == len(self.tuples['d'])== len(self.tuples['gr'])):
				#	pdb.set_trace()
		except:
			print("ISSUE")


		self.tuples['s'] = None
		self.tuples['ns'] = None
		self.tuples['a'] = None
		self.tuples['pc'] = None
		self.tuples['r'] = None
		self.tuples['d'] = None
		self.tuples['gr'] = None

		count = 0
		if len(self.s) > self.capacity:
			self.s = self.s[-self.capacity:, :]
			self.ns = self.ns[-self.capacity:, :]
			self.a = self.a[-self.capacity:, :]
			self.policy_choice = self.policy_choice[-self.capacity:, :]
			self.r = self.r[-self.capacity:, :]
			self.done = self.done[-self.capacity:, :]
			self.global_reward = self.global_reward[-self.capacity:, :]
		# pdb.set_trace()
		'''

	def __len__(self):
		return len(self.s)


	def sample(self, batch_size, pr_rew=0.0, pr_global=0.0, type=None):

		#try: ind = random.sample(range(len(self.sT)), batch_size)
		#except: print(self.buffer_type, "Failure")
		#print(self.buffer_type, "Success")
		#length = min(len(self.s), len(self.ns), len(self.a), len(self.policy_choice), len(self.r), len(self.done), len(self.global_reward))
		if type == "policy_selector": # sample the last k samples
			#ind = random.sample(range(len(self.s)-20*batch_size, len(self.s)), batch_size) # sample from the latest experiences # todo: change to 50

			#ind = random.sample(range(int(length/2), length), batch_size) # sample from the latest experiences # todo: change to 50
			if (len(self.sT) <= 40000):
				ind = random.sample(range(len(self.sT)-10), batch_size)
			else:
				ind = random.sample(range(len(self.sT[-20000:])), batch_size)

			#ind = random.sample(range(int(len(self.sT)/2), len(self.s)), batch_size)

		else: #ind = random.sample(range(length), batch_size)
			ind = random.sample(range(len(self.sT)-10), batch_size)

		if pr_global != 0.0 or pr_rew !=0:
			#pdb.set_trace()
			num_r = int(pr_rew * batch_size);
			num_global = int(pr_global * batch_size)
			ind_r = random.sample(self.top_r, num_r)
			ind_global = random.sample(self.top_g, num_global)

			ind = ind[num_r + num_global:] + ind_r + ind_global

		'''
		try:

			#return self.s[ind], self.ns[ind], self.a[ind], self.policy_choice[ind], self.r[ind], self.done[ind], self.global_reward[ind]
			return self.sT[ind], self.nsT[ind], self.aT[ind], self.policy_choiceT[ind], self.rT[ind], self.doneT[ind], self.global_rewardT[ind]

		except:
				print("Here comes an issue")
				import pdb
				pdb.set_trace()
		'''
		# no need to use the tensorified data
		return self.sT[ind], self.nsT[ind], self.aT[ind], self.policy_choiceT[ind], self.rT[ind], self.doneT[ind], self.global_rewardT[ind]


	def tensorify(self):
		self.refresh()

		if self.__len__() > 1:
			self.sT = torch.tensor(np.vstack(self.s))
			self.nsT = torch.tensor(np.vstack(self.ns))
			self.aT = torch.tensor(np.vstack(self.a))
			self.policy_choiceT = torch.tensor(np.vstack(self.policy_choice))
			self.rT = torch.tensor(np.vstack(self.r))
			self.doneT = torch.tensor(np.vstack(self.done))
			self.global_rewardT = torch.tensor(np.vstack(self.global_reward))

			#if self.buffer_type == 'Primitive_agents':
			#	self.rT = torch.tensor(np.vstack(self.r))

			#else:
			#	self.policy_choiceT = torch.tensor(np.vstack(self.policy_choice))
			if self.buffer_gpu:
				self.sT = self.sT.cuda()
				self.nsT = self.nsT.cuda()
				self.aT= self.aT.cuda()
				self.policy_choiceT = self.policy_choiceT.cuda()
				self.rT = self.rT.cuda()
				self.doneT = self.doneT.cuda()
				self.global_reward = self.global_rewardT.cuda()

				#if self.buffer_type == 'Primitive_agents':
				#	self.rT = self.rT.cuda()
				#else:

				#	self.policy_choice = self.policy_choiceT.cuda()

			# Prioritized indices update
			#if self.buffer_type == 'Primitive_agents':
			#	self.top_r = list(np.argsort(np.vstack(self.r), axis = 0)[-int(len(self.s)/10):])


			self.top_r = list(np.argsort(np.vstack(self.r), axis=0)[-int(len(self.s) / 10):])
			self.top_g = list(np.argsort(np.vstack(self.global_reward), axis = 0)[-int(len(self.s)/10):])

			# update stats
			#if self.buffer_type == 'Primitive_agents':
			#	compute_stats(self.rT, self.rstats)

			compute_stats(self.rT, self.rstats)
			compute_stats(self.global_rewardT, self.gstats)



