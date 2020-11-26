from core import mod_utils as utils
import numpy as np, random, sys
import torch
import pdb


def rollout_worker(args, id, type, # args, id, type is for population
                   task_pipe,     # task pipe
                   result_pipe,   # result_pipe
                   centralized_buffer_dict, # primitives_centralized buffer
                   primitive_policies_bucket, # primitive policies bucket
                   policy_selector_bucket,  # policy selector model
                   store_transitions, # True/ False, False for primitive agents, and test agents
                   random_baseline_policy_selector, # if its a random baseline
                   random_baseline_primitives,
                   epsilon,
                   reward_type_idx): # reward type index

    if type == 'test':
        NUM_EVALS = args.overall_num_test
        #NUM_EVALS = 4

    elif type == 'policy_selector_pg':
        NUM_EVALS = args.policy_selector_rollout_size
        #NUM_EVALS = 4

    elif type == 'primitive_policies':
        NUM_EVALS = args.primitives_rollout_size
        #NUM_EVALS = 1#2


    elif type == 'policy_selector_evo':
        #NUM_EVALS = args.policy_selector_popn_size
        NUM_EVALS = 10

    elif type == 'primitive_test':
        NUM_EVALS = args.primitives_test_rollouts

    else:
        sys.exit('Incorrect type for rollout worker')


    if args.config.env_choice == 'rover_heterogeneous':
        from envs.env_wrapper import RoverHeterogeneousSequential
        #env = RoverHeterogeneousSequential(args, NUM_EVALS, reward_type_idx, type)

        from envs.env_wrapper import SequentialPOIRD
        env = SequentialPOIRD(args, NUM_EVALS, reward_type_idx, type)

    elif args.config.env_choice == 'rover_homogeneous':
        from envs.env_wrapper import RoverDomainPython
        env = RoverDomainPython(args, NUM_EVALS)

    else:
        sys.exit('Incorrect env type')

    ## for environment seeds
    np.random.seed(id); random.seed(id)


    # This is one rollout
    while True:

        signal = task_pipe.recv()
        teams_blueprint = signal[0]

        if teams_blueprint == 'TERMINATE': exit(0)  # kill the process and exit the program

        if type == 'policy_selector_pg':
            epsilon = signal[1]
            print("$$$$$$$$$$ EPSILON NOW for DQN $$$$$$$$$$", epsilon)

        # rollout trajectory to store experiences into the buffer
        rollout_trajectory_policy_selector = []


        if type == 'test':
            reward_type_selected = [[None for _ in range(args.config.ep_len)] for _ in  range(args.config.num_agents)]
            team = policy_selector_bucket
            primitive_team = primitive_policies_bucket

        elif type == 'policy_selector_pg' or type == 'policy_selector_evo':
            import time
            time_start = time.time()
            team = policy_selector_bucket
            primitive_team = primitive_policies_bucket

        elif type == 'primitive_policies' or type == 'primitive_test':
            #pdb.set_trace()
            primitive_team = primitive_policies_bucket

        rollout_trajectory_primitives = [[] for _ in range(args.config.num_agents)]  # for primitive agent's learning

        '''
        rollout_trajectory_primitives = [{} for _ in range(args.config.num_agents)]  # for primitive agent's learning
        for agent_id in range(args.config.num_agents):
            rollout_trajectory_primitives[agent_id]['s'] = [];  rollout_trajectory_primitives[agent_id]['ns'] = [];  rollout_trajectory_primitives[agent_id]['a'] = [];
            rollout_trajectory_primitives[agent_id]['pc'] = [];  rollout_trajectory_primitives[agent_id]['r'] = [];  rollout_trajectory_primitives[agent_id]['d'] = [];
            rollout_trajectory_primitives[agent_id]['gr'] = []
        '''

        fitness = [None for _ in range(NUM_EVALS)]
        frame = 0.0

        local_reward = [[[0.0 for _ in range(len( args.config.reward_types))] for _ in range(NUM_EVALS)] for _ in range(args.config.num_agents)]

        ## reset environment
        joint_state = env.reset();
        joint_state = utils.to_tensor(np.array(joint_state))

        episode_time_step = -1;

        while True:
            episode_time_step+=1 # starts from 1

            if random_baseline_policy_selector:
                policy_choices = []
                for _ in range(args.config.num_agents):
                    policy_choices.append(torch.randint(0, len(args.config.reward_types) + args.cardinal_actions, (NUM_EVALS,))) # taking the max reward type chosen per episode
                    #policy_choices.append(torch.randint(0, 1, (NUM_EVALS,)))  #

                policy_choices = np.array(torch.stack(policy_choices))

            elif type == 'primitive_policies' or type == 'primitive_test':
                policy_choices = torch.zeros([args.config.num_agents, joint_state.size()[1]])

                for i in range(joint_state.size()[1]):
                    policy_choices[:, i] = reward_type_idx
                policy_choices = np.array(policy_choices)
                policy_choices = np.reshape(policy_choices, (args.config.num_agents, -1, 1))

            elif type == 'policy_selector_pg' or type == 'test':
                '''
                policy_choices = team[0][0].policy_selector(joint_state[0, :, :]) # for agent 0 #None# torch.tensor(0)
                if args.config.num_agents > 1:
                    for agent_id in range(1, args.config.num_agents):
                        policy_choices = torch.stack([policy_choices, team[agent_id][0].policy_selector(joint_state[agent_id, :, :])], dim =0)

                policy_choices = np.array(policy_choices)
                '''

                #if type == 'test': pdb.set_trace()
                if type == 'policy_selector_pg':
                    policy_choices = [team[agent_id][0].policy_selector(joint_state[agent_id, :, :], epsilon) for
                                        agent_id in range(args.config.num_agents)]  # will give different policies for each agent
                    policy_choices = np.array(torch.stack(policy_choices))

                elif type == 'test':
                    policy_choices = [team[agent_id].policy_selector(joint_state[agent_id, :, :], 0) for
                                      agent_id in range(args.config.num_agents)]  # will give different policies for each agent
                    policy_choices = np.array(torch.stack(policy_choices))

                    for agent_id in range(args.config.num_agents):
                        average_reward_type = []
                        for reward_id in range(len(args.config.reward_types)+args.cardinal_actions):
                            average_reward_type.append(policy_choices[agent_id].tolist().count(
                                reward_id))  # contains number of times each reward occurs at each time step

                        reward_type_selected[agent_id][episode_time_step] = average_reward_type

            elif type == 'policy_selector_evo':
                '''
                policy_choices = team[0][teams_blueprint[0]].policy_selector(joint_state[0, :, :], epsilon)  # for 1st agent
                if args.config.num_agents > 1:
                    for agent_id, pop_id in enumerate(teams_blueprint[1: ]):
                        policy_choices = torch.stack([policy_choices, team[agent_id + 1][0].policy_selector(joint_state[agent_id + 1, :, :])], dim =0)

                policy_choices = np.array(policy_choices)
                '''
                policy_choices = [team[agent_id][pop_id].policy_selector(joint_state[agent_id, :, :], 0) for agent_id, pop_id in enumerate(teams_blueprint)]

                policy_choices = np.array(torch.stack(policy_choices))

            if random_baseline_primitives:
                joint_action = [np.random.random((NUM_EVALS, args.state_dim))for _ in range(args.config.num_agents)]
                joint_action = np.reshape(joint_action, (args.config.num_agents, -1, args.agent_action_dim))

            elif type == 'test' or type == 'policy_selector_evo':

                joint_action = []
                joint_action_choice = []
                for i in range(args.config.num_agents):
                    action = []
                    action_choice = []
                    for universe_id in range(len(policy_choices[i])):
                        # print(policy_choices[:, i])
                        #reward_type = args.config.reward_types[policy_choices[i][universe_id].item()]
                        #action.append(primitive_policies_bucket[reward_type][0].clean_action(joint_state[i, universe_id, :]).detach().numpy())
                        reward_type = int(policy_choices[i][universe_id])

                        if reward_type == len(args.config.reward_types):
                            action.append(np.array([[-1, 0]], dtype='float32')) # go left
                            action_choice.append('L')
                        elif reward_type == len(args.config.reward_types) + 1:
                            action.append(np.array([[1, 0]], dtype='float32')) # go right
                            action_choice.append('R')
                        elif reward_type == len(args.config.reward_types) + 2:
                            action.append(np.array([[0, -1]], dtype='float32')) # go down
                            action_choice.append('D')
                        elif reward_type == len(args.config.reward_types) + 3:
                            action.append(np.array([[0, 1]], dtype='float32')) # go up
                            action_choice.append('U')
                        else:
                            action.append(primitive_policies_bucket[reward_type][0].clean_action(joint_state[i, universe_id, :]).detach().numpy())
                            action_choice.append(None)

                        #pdb.set_trace()

                    joint_action.append(action)
                    joint_action_choice.append(action_choice)

                joint_action_choice = np.reshape(joint_action_choice, (args.config.num_agents, -1))
                joint_action = np.reshape(joint_action, (args.config.num_agents, -1, args.agent_action_dim))

            elif type == 'policy_selector_pg': # exploration at both level
                joint_action = []
                joint_action_choice = []
                for i in range(args.config.num_agents):
                    action = []
                    action_choice = []
                    for universe_id in range(len(policy_choices[i])):
                        #reward_type = args.config.reward_types[policy_choices[i][universe_id].item()]
                        #action.append(primitive_policies_bucket[reward_type][0].noisy_action(joint_state[i, universe_id, :]).detach().numpy())
                        reward_type = int(policy_choices[i][universe_id])

                        if reward_type == len(args.config.reward_types):
                            action.append(np.array([[-1, 0]], dtype='float32')) # go left
                            action_choice.append('L')
                        elif reward_type == len(args.config.reward_types) + 1:
                            action.append(np.array([[1, 0]], dtype='float32')) # go right
                            action_choice.append('R')
                        elif reward_type == len(args.config.reward_types) + 2:
                            action.append(np.array([[0, -1]], dtype='float32')) # go down
                            action_choice.append('D')
                        elif reward_type == len(args.config.reward_types) + 3:
                            action.append(np.array([[0, 1]], dtype='float32')) # go up
                            action_choice.append('U')
                        else:
                            action.append(primitive_policies_bucket[reward_type][0].noisy_action(joint_state[i, universe_id, :]).detach().numpy())
                            action_choice.append(None)

                    joint_action.append(action)
                    joint_action_choice.append(action_choice)

                joint_action_choice = np.reshape(joint_action_choice, (args.config.num_agents, -1))
                joint_action = np.reshape(joint_action, (args.config.num_agents, -1, args.agent_action_dim))

            elif type == 'primitive_policies': # for primitives rollouts
                joint_action = [primitive_team[0].noisy_action(joint_state[i, :]).detach().numpy() for i in range(args.config.num_agents)]
                joint_action_choice = [[None for _ in range(args.config.num_agents)] for _ in range(NUM_EVALS)]

            else: #primitive_test'
                #print('#########################', type, '#######################')
                joint_action = [primitive_team[0].clean_action(joint_state[i, :]).detach().numpy() for i in range(args.config.num_agents)]
                joint_action_choice = [[None for _ in range(args.config.num_agents)] for _ in range(NUM_EVALS)]

            joint_action_choice = np.reshape(joint_action_choice, (args.config.num_agents, -1))
            joint_action = np.reshape(joint_action, (args.config.num_agents, -1, args.agent_action_dim))
            policy_choices = np.reshape(policy_choices, (args.config.num_agents, -1, 1))

            # policy_choices = torch.stack(policy_choices)  # concatenate the list of tensors to one tensor
            # policy_choices = policy_choices.unsqueeze(2)
            # policy_choices = np.reshape(policy_choices, (-1, args.config.num_agents))
            # policy_choices = torch.cat(policy_choices[0], dim = -1)

            #next_state, reward, done, global_reward = env.step(joint_action, policy_choices)  # domain knowledge comes here, where we need access to the simulator in these cases
            next_state, reward, done, global_reward = env.step(joint_action, joint_action_choice)  # domain knowledge comes here, where we need access to the simulator in these cases

            next_state = utils.to_tensor(np.array(next_state))
            #policy_choices = utils.to_tensor(np.array(policy_choices))
            if args.config.env_choice == 'rover_homogeneous': # for dimensions
                reward = np.reshape(reward, (reward.shape[0], reward.shape[1], -1))

            local_reward += reward

            #local_reward = [[[local_reward[i][j][k] + reward[i][j][k] for k in range(len(local_reward[0][0]))] for j in range(len(local_reward[0]))] for i in range(len(local_reward))]

            ## get global reward as fitness
            for i, grew in enumerate(global_reward):  # #for evo, its one for each pop, so list=pop size, for pg, list length = rollout size
                if grew != None:
                    fitness[i] = grew  # adding the global reward for each rollout

            if store_transitions:
                # For Policy Selector, push the experiences into both: agent's own buffer as well as Policy selector buffer
                for agent_id in range(args.config.num_agents):
                    for universe_id in range(NUM_EVALS):
                        if not done[universe_id]:
                            # add individual components into agent's own replay buffer

                            rollout_trajectory_primitives[agent_id].append(
                                [np.expand_dims(utils.to_numpy(joint_state)[agent_id, universe_id, :], 0), # agent's own state
                                 np.expand_dims(utils.to_numpy(next_state)[agent_id, universe_id, :], 0),  # agent's own next state
                                 np.expand_dims(joint_action[agent_id, universe_id, :], 0), # low level actions from primitives
                                 np.expand_dims(policy_choices[agent_id, universe_id, :], 0), # high level actions
                                 np.expand_dims(np.array([reward[agent_id, universe_id, :]], dtype="float32"), 0), # agent's own local reward
                                 np.expand_dims(np.array([done[universe_id]], dtype="float32"), 0), # done flag
                                 universe_id])

                            '''
                            rollout_trajectory_primitives[agent_id]['s'].append(joint_state[agent_id, universe_id, :].unsqueeze(0))
                            rollout_trajectory_primitives[agent_id]['ns'].append(next_state[agent_id, universe_id, :].unsqueeze(0))
                            rollout_trajectory_primitives[agent_id]['a'].append(utils.to_tensor(joint_action[agent_id, universe_id, :]).unsqueeze(0))
                            rollout_trajectory_primitives[agent_id]['pc'].append(utils.to_tensor(policy_choices[agent_id, universe_id, :]).unsqueeze(0))
                            rollout_trajectory_primitives[agent_id]['r'].append(utils.to_tensor(reward[agent_id, universe_id, :]).unsqueeze(0))
                            rollout_trajectory_primitives[agent_id]['d'].append(utils.to_tensor(np.array([done[universe_id]])).unsqueeze(0))
                            rollout_trajectory_primitives[agent_id]['gr'].append(universe_id)
                            '''
                            # type for the experience

            joint_state = next_state

            if type == 'policy_selector_pg' or 'policy_selector_evo':
                frame += NUM_EVALS

            if type == type == "primitive_policies": # todo: remove this
                frame += NUM_EVALS


            # DONE FLAG IS RECEIEVED for all envs
            if sum(done) == len(done):
                # Push experiences to main
                if store_transitions:
                    # Push experiences to main
                     # store individual states in agent's own buffer

                    # send this whole rollout_trajectory_primitives as whole, instead of updating the buffer here itself


                    for agent_id, buffer in enumerate(centralized_buffer_dict):
                        for entry in rollout_trajectory_primitives[agent_id]:
                            temp_global_reward = fitness[entry[6]]
                            entry[6] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)  # changing universe id with fitness value
                            buffer.append(entry)

                    '''
                    for agent_id in range(args.config.num_agents):
                       for entry in rollout_trajectory_primitives[agent_id]:
                           temp_global_reward = fitness[entry[6]]
                           entry[6] = utils.to_tensor(np.array([temp_global_reward])).unsqueeze(0)  # changing universe id with fitness value
                           #buffer.tuples.append(entry)
                    '''
                    '''
                    for agent_id in range(args.config.num_agents):
                       for id, entry in enumerate(rollout_trajectory_primitives[agent_id]['gr']):
                           temp_global_reward = fitness[entry]
                           rollout_trajectory_primitives[agent_id]['gr'][id] = utils.to_tensor(np.array([temp_global_reward])).unsqueeze(0)  # changing universe id with fitness value
                           #buffer.tuples.append(entry)

                    for agent_id, agent_entry in enumerate(centralized_buffer_dict):
                        agent_entry['s'] = rollout_trajectory_primitives[agent_id]['s']
                        agent_entry['ns'] = rollout_trajectory_primitives[agent_id]['ns']
                        agent_entry['a'] = rollout_trajectory_primitives[agent_id]['a']
                        agent_entry['pc'] = rollout_trajectory_primitives[agent_id]['pc']
                        agent_entry['r'] = rollout_trajectory_primitives[agent_id]['r']
                        agent_entry['d'] = rollout_trajectory_primitives[agent_id]['d']
                        agent_entry['gr'] = rollout_trajectory_primitives[agent_id]['gr']

                       #centralized_buffer[agent_id].add(rollout_trajectory_primitives[agent_id])
                    '''

                break

            # if type == 'test' and args.config.env_choice:
            #    env.render();


        if (type == "primitive_test") or type == "test":
            env.render()

        if type == "primitive_policies" or type == "primitive_test":
            #pdb.set_trace()

            episodic_reward = [[] for _ in range(args.config.num_agents)]

            for agent_id in range(args.config.num_agents):
                episodic_reward[agent_id] = []
                for universe_id in range(NUM_EVALS):
                    episodic_reward[agent_id].append(local_reward[agent_id][universe_id][reward_type_idx])

            #print("DONE", len(primitives_centralized_buffer[0].tuples))
            result_pipe.send([teams_blueprint, [fitness], [episodic_reward], frame])

        elif type == "test":
            rewards_selected = []
            for i in range(args.config.num_agents):
                rewards_selected.append([np.argmax(reward_type_selected[i][k]) for k in range(args.config.ep_len)]) # taking the max reward type chosen per episode
            result_pipe.send([teams_blueprint, [fitness], None, rewards_selected])
        else:

            #pdb.set_trace()
            #print(type, time.time() - time_start)
            #result_pipe.send([teams_blueprint, [fitness], local_reward, frame, rollout_trajectory_primitives])
            result_pipe.send([teams_blueprint, [fitness], local_reward, frame])


