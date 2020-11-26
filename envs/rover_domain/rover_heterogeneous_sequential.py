import random, sys
from random import randint
import numpy as np
import math
import matplotlib.pyplot as plt

class RoverDomainHeterogeneousSequential:
    '''
    This is for heterogeneous POIs, POIs of different types, and homogeneous agents
    '''

    def __init__(self, args, reward_type_idx, type, env_id):

        self.args = args
        self.task_type = args.env_choice
        self.harvest_period = args.harvest_period # set as 1 for all except as 3 for env as trap
        self.reward_type_idx = reward_type_idx
        self.type = type
        self.env_id = env_id

        self.coupling = self.args.coupling

        '''
        if self.reward_type_idx == -1: # for overall complete architecrure
            self.coupling = self.args.coupling # actual coupling specified
        else: self.coupling = [1, 1] # for primitives, or primitive tests, so that they learn to go to POI
        '''

        if reward_type_idx !=-1:
            self.reward_type = self.args.reward_types[reward_type_idx]

        #Gym compatible attributes
        self.observation_space = np.zeros((1, int(2*360 / self.args.angle_res)+1))
        self.action_space = np.zeros((1, 2))

        self.istep = 0 #Current Step counter
        self.done = False

        # Initialize POI containers tha track POI position and status
        self.poi_pos = [[None, None] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][x, y] coordinate

        self.poi_status = {}  # distance of all rovers (trucks+uavs) from POIs
        self.poi_value = {}
        for i in range(self.args.num_poi_types):
            self.poi_status[i] = [self.harvest_period for _ in range(self.args.num_poi_A)]  # assuming all number of all of types of POIs is same
            self.poi_value[i] = [1.0 for _ in range(self.args.num_poi_A)]

        #self.poi_value = [float(i+1) for i in range(self.args.num_poi)]  # FORMAT: [poi_id][value]?
        #self.poi_value = [1.0 for _ in range(self.args.num_poi)]
        self.poi_visitor_list = [[] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][visitors]?

        # Initialize rover pose container
        self.rover_pos = [[0.0, 0.0, 0.0] for _ in range(self.args.num_agents)]  # FORMAT: [rover_id][x, y, orientation] coordinate with pose info
        self.rover_vel = [[0.0, 0.0] for _ in range(self.args.num_agents)]

        #Local Reward computing methods
        self.rover_closest_poi = {}
        for i in range(self.args.num_agents):
            self.rover_closest_poi[i] = [self.args.dim_x*2 for _ in range( self.args.num_poi_types)] # distance of all rovers (trucks+uavs) from POIs

        self.uav_closest_rover = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]
        self.uav_closest_uav = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]    # distance of all uavs from a UAV
        self.rover_closest_uav = [self.args.dim_x * 2 for _ in range(self.args.num_agents)] # distance of trucks from UAVs
        self.rover_closest_rover = [self.args.dim_x * 2 for _ in range(self.args.num_agents)] # distance of fire trucks from other fire trucks

        #self.rover_closest_poi_2 = [self.args.dim_x*2 for _ in range(self.args.num_agents)]
        #self.rover_farthest_poi_2 = [0  for _ in range(self.args.num_agents)] # distamce to all rovers from POIs (UAVs + Trucks)

        #self.cumulative_local = [0.0 for _ in range(self.args.num_agents)]
        self.cumulative_local = [[0.0 for _ in range(len(self.args.reward_types))] for _ in range(self.args.num_agents)]

        #Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = [[] for _ in range(self.args.num_agents)] # FORMAT: [rover_id][timestep][x, y]
        self.action_seq = [[] for _ in range(self.args.num_agents)] # FORMAT: [timestep][rover_id][action]


    def reset(self):
        self.done = False
        self.reset_poi_pos()
        self.reset_rover_pos()
        self.rover_vel = [[0.0, 0.0] for _ in range(self.args.num_agents)]
        #self.poi_value = [float(i+1) for i in range(self.args.num_poi)]
        self.rover_closest_poi = {}
        for i in range(self.args.num_agents):
            self.rover_closest_poi[i] = [self.args.dim_x * 2 for _ in range(self.args.num_poi_types)]  # distance of all rovers (trucks+uavs) from POIs

        self.uav_closest_rover = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]
        self.uav_closest_uav = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]  # distance of all uavs from a UAV
        self.rover_closest_uav = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]  # distance of trucks from UAVs
        self.rover_closest_rover = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]  # distance of fire trucks from other fire trucks

        #self.cumulative_local = [0 for _ in range(self.args.num_agents)]
        self.cumulative_local = [[0.0 for _ in range(len(self.args.reward_types))] for _ in range(self.args.num_agents)]

        self.poi_status = {}  # distance of all rovers (trucks+uavs) from POIs
        for i in range(self.args.num_poi_types):
            self.poi_status[i] = [self.harvest_period for _ in range(self.args.num_poi_A)]  # assuming all number of all of types of POIs is same
            self.poi_value[i] = [1.0 for _ in range(self.args.num_poi_A)]

        self.poi_visitor_list = [[] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][visitors]?
        self.rover_path = [[] for _ in range(self.args.num_agents)]
        self.action_seq = [[] for _ in range(self.args.num_agents)]
        self.istep = 0

        return self.get_joint_state()


    def step(self, joint_action, action_choice): # returns all actions

        #If done send back dummy trasnsition
        if self.done:
            dummy_state, dummy_reward, done, info = self.dummy_transition()
            return dummy_state, dummy_reward, done, info

        self.istep += 1

        joint_action = joint_action.clip(-1.0, 1.0)

        for rover_id in range(self.args.num_agents):
            # todo: different action space for both rovers and POIs
            multiplier = 1.0

            '''
            if self.args.action_space == "different":
                rover_type = int(0) if rover_id < self.args.num_uavs else int(1)

                if rover_type == 0:  # uav
                    multiplier = 1.5
            '''

            if action_choice[rover_id] == 'L' or action_choice[rover_id] == 'R' or action_choice[rover_id] == 'U' or action_choice[rover_id] == 'D':
                self.rover_pos[rover_id][0] += 0.5 * multiplier * joint_action[rover_id][0]
                self.rover_pos[rover_id][1] += 0.5 * multiplier * joint_action[rover_id][1]

                self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1], self.rover_pos[rover_id][2]))
                self.action_seq[rover_id].append([0.5*multiplier*joint_action[rover_id][0], 0.5*multiplier*joint_action[rover_id][1]])

            else:

                magnitude = 0.5*(joint_action[rover_id][0]+1) # [-1,1] --> [0,1]
                self.rover_vel[rover_id][0] += multiplier * magnitude

                joint_action[rover_id][1] /= 2.0 #Theta (bearing constrained to be within 90 degree turn from heading)
                self.rover_vel[rover_id][1] += joint_action[rover_id][1]

                #Constrain
                '''
                self.rover_vel[rover_id][0] = max(0, min(1, self.rover_vel[rover_id][0]))
                self.rover_vel[rover_id][1] = max(-0.5, min(0.5, self.rover_vel[rover_id][1]))
                '''
                if self.rover_vel[rover_id][0] < 0: self.rover_vel[rover_id][0] = 0.0
                elif self.rover_vel[rover_id][0] > 1: self.rover_vel[rover_id][0] = 1.0

                if self.rover_vel[rover_id][1] < 0.5: self.rover_vel[rover_id][0] = 0.5
                elif self.rover_vel[rover_id][1] > 0.5: self.rover_vel[rover_id][0] = 0.5

                theta = self.rover_vel[rover_id][1] * 180 + self.rover_pos[rover_id][2]
                if theta > 360: theta -= 360
                elif theta < 0: theta += 360
                #self.rover_pos[rover_id][2] = theta

                #Update position
                x = self.rover_vel[rover_id][0] * math.cos(math.radians(theta))
                y = self.rover_vel[rover_id][0] * math.sin(math.radians(theta))

                self.rover_pos[rover_id][0] += x
                self.rover_pos[rover_id][1] += y

                #Log
                self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1], self.rover_pos[rover_id][2]))
                self.action_seq[rover_id].append([magnitude, joint_action[rover_id][1]*180])


        #Compute done
        self.done = int(self.istep >= self.args.ep_len or sum(self.poi_status) == 0)

        #info
        global_reward = None
        if self.done: global_reward = self.get_global_reward()

        return self.get_joint_state(), self.get_local_reward(), self.done, global_reward


    def reset_poi_pos(self):

        start = 0.0;
        end = self.args.dim_x - 1.0
        #rad = int(self.args.dim_x/(2*(math.sqrt(10)))) #         int(self.args.dim_x / math.sqrt(3) / 2.0)
        rad = int(self.args.dim_x / (2 * (math.sqrt(10)))) + 2
        center = int((start + end) / 2.0)

        rad = 8

        if(self.args.EVALUATE_only): #for testing
            for i in range(self.args.num_poi):
                x = randint(start, center - rad - 1)
                y = randint(start, end)
                self.poi_pos[i] = [x, y]
        else:
            if self.args.poi_rand: #Random

                for i in range(self.args.num_poi):

                    poi_type = 0
                    if self.args.num_poi_A <=  i < self.args.num_poi_A + self.args.num_poi_B: poi_type = 1
                    if self.args.num_poi_A + self.args.num_poi_B <=  i < self.args.num_poi: poi_type = 2

                    if poi_type % 3 == 0:
                        x = randint(center - rad - 1, center)
                        y = randint(center - rad - 1, center)

                    elif poi_type % 3 == 1:
                        x = randint(center, center + rad + 1)
                        y = randint(center , center + rad + 1)

                    elif poi_type % 3 == 2:
                        x = randint(center - rad - 1, center)
                        if random.random() < 0.5:
                            y = randint(center, center + rad + 1)
                        else:
                            y = randint(center-rad -1, center)

                    self.poi_pos[i] = [x, y]

                    '''
                    if i % 3 == 0:
                        x = randint(start, center - rad - 1)
                        y = randint(start, end)
                    elif i % 3 == 1:
                        x = randint(center + rad + 1, end)
                        y = randint(start, end)
                    elif i % 3 == 2:
                        x = randint(center - rad, center + rad)
                        if random.random() < 0.5:
                            y = randint(start, center - rad - 1)
                        else:
                            y = randint(center + rad + 1, end)

                    self.poi_pos[i] = [x, y]
                    '''

                '''
                for i in range(self.args.num_poi):
                    #if self.type == "test" or self.type == "primitive_test":
                    random.seed(self.env_id + i)  # keep it fixed for test env
                    if i % 4 == 2:
                        x = randint(start+1, center-rad-1)
                        y = randint(start+1, center-rad-1)

                    elif i % 4 == 0:
                        x = randint(center + rad + 1, end - 1)
                        y = randint(start + 1, center - rad -1)

                    elif i % 4 == 1:
                        x = randint(center + rad + 1, end - 1)
                        y = randint(center + rad + 1, end - 1)

                    else:
                        x = randint(start + 1, center - rad - 1)
                        y = randint(center + rad + 1, end - 1)

                    self.poi_pos[i] = [x, y]
                '''

            else: #Not_random
                for i in range(self.args.num_poi):
                    if i % 4 == 0:
                        x = 5 #int((start  + center)/2)  #start + int(i/2) #randint(start, center - rad - 1)
                        y = 5 #center #start + int(i/2)
                    elif i % 4 == 1:
                        x =  5 #center #end - int(i/2) #randint(center + rad + 1, end)
                        y =  15#int((end  - center)/2)##randint(start, end)
                    elif i % 4 == 2:
                        x = 15#int((end  - center)/2)  #start+ int(i/2) #randint(center - rad, center + rad)
                        y = 5 #int((start  + center)/2)#end - int(i/2) #randint(start, center - rad - 1)
                    else:
                        x = 15 #int((end  - center)/2)#end - int(i/2) #randint(center - rad, center + rad)
                        y = 15#int((end  - center)/2)#end - int(i/2) #randint(center + rad + 1, end)
                    self.poi_pos[i] = [x, y]


    def reset_rover_pos(self):
        start = 1.0; end = self.args.dim_x - 1.0
        rad = int(self.args.dim_x/(2*(math.sqrt(10)))) #10% area in the center for Rovers
        center = int((start + end) / 2.0)

        # Random Init
        #rad = 5

        lower = center - rad
        upper = center + rad

        # Random Init
        lower = center - rad
        upper = center + rad
        for i in range(self.args.num_agents):
            x = randint(lower, upper)
            y = randint(lower, upper)
            self.rover_pos[i] = [x, y, 0.0]

        '''
        for i in range(self.args.num_agents):
            #if type == "test" or type == "primitive_test":
            random.seed(self.env_id + i)  # keep it fixed for test env

            if i % 4 == 2:
                x = randint(center, upper)  # todo: need to change this -2
                y = randint(lower, center)  # todo: need to change this -2

            if i % 4 == 3:
                x = randint(lower, center)  # todo: need to change this -2
                y = randint(center, upper)  # todo: need to change this -2

            if i % 4 == 0:
                x = randint(lower, center)  # todo: need to change this -2
                y = randint(lower, center)  # todo: need to change this -2
                #x = randint(start + 1, center - rad - 1)
                #y = randint(start + 1, center - rad - 1)

            if i % 4 == 1:
                #x = randint(start + 1, center - rad - 1)
                #y = randint(center, end-1)
                x = randint(center, upper)  # todo: need to change this -2
                y = randint(center, upper)  # todo: need to change this -2

            # x = randint(lower, upper)  # todo: need to change this -2
            # y = randint(lower, upper)  # todo: need to change this -2
            # x = randint(start+1, end-1) # todo: need to change this -2
            # y = randint(start+1, end-1) # todo: need to change this -2
            self.rover_pos[i] = [x, y, 0.0]
            '''


        '''
        if self.reward_type == 'go_to_poi':

            # Random Init
            lower = center - rad
            upper = center + rad

            for i in range(self.args.num_agents):
                x = randint(lower, upper-2) # todo: need to change this -2
                y = randint(lower, upper-2) # todo: need to change this -2
                #x = randint(lower, upper)  # todo: need to change this -2
                #y = randint(lower, upper)  # todo: need to change this -2
                #x = randint(start+1, end-1) # todo: need to change this -2
                #y = randint(start+1, end-1) # todo: need to change this -2
                self.rover_pos[i] = [x, y, 0.0]

        else:

            # Random Init
            lower = center - int(end/3)
            upper = center + int(end/3)

            self.rover_pos[0] = [10, 15, 0];
            self.rover_pos[1] = [15, 10, 0];
            self.rover_pos[2] = [15, 15, 0];
            self.rover_pos[3] = [10, 10, 0];


            '''

        #self.rover_pos[0] = [11, 9, 0.0] #todo: for testing only
        #self.rover_pos[2][0] = self.rover_pos[1][0] + 5 # initialize truck to be very close to truck, or may at the same location itself
        #print(self.rover_pos)


    def get_joint_state(self):
        joint_state = []
        for rover_id in range(self.args.num_agents): # for each rover, check each POI and other rovers, whether that POI is in that range
            self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]; self_orient = self.rover_pos[rover_id][2]

            rover_state = [0.0 for _ in range(int(360 *self.args.num_agent_types/self.args.angle_res))] # added for heterogeneous rovers (different types of rovers)
            #rover_state = [0.0 for _ in range(int(360 / (self.args.angle_res)))]
            poi_state = [0.0 for _ in range(int(360*self.args.num_poi_types / self.args.angle_res))]
            temp_poi_dist_list = [[] for _ in range(int(360 *self.args.num_poi_types/ self.args.angle_res))]
            temp_rover_dist_list = [[] for _ in range(int(360 * self.args.num_agent_types/self.args.angle_res))]
            #temp_rover_dist_list = [[] for _ in range(int(360 / (self.args.angle_res)))]

            #rover_type_ref = int(rover_id / self.args.num_uavs)
            rover_type_ref = int(0) if rover_id < self.args.num_uavs else int(1)
            # Log all distance into brackets for POIs

            for poi_id in range(self.args.num_poi): # for each poi
                #if status == 0 and self.reward_type_idx == -1: continue #If accessed ignore (only for global tasks)
                # POI type
                poi_type = 0
                if self.args.num_poi_A <= poi_id < self.args.num_poi_A + self.args.num_poi_B: poi_type = 1
                if self.args.num_poi_A + self.args.num_poi_B <= poi_id < self.args.num_poi: poi_type = 2

                if self.poi_status[poi_type][poi_id if poi_type == 0 else poi_id % self.args.num_poi_A] == 0: continue #If accessed ignore (only for global tasks), joint state should be same

                angle, dist = self.get_angle_dist(self_x, self_y, self.poi_pos[poi_id][0], self.poi_pos[poi_id][1])

                ## todo: this is added for long_range_lidar of truck
                if self.args.config == 'fire_truck_uav_long_range_lidar':
                    try: bracket = int(angle / self.args.angle_res)
                    except: bracket = 0


                    if (rover_type_ref == 1) and bracket == 0: # as 0 is the longest range lidar in truck
                        if dist > self.args.long_range: continue  # Observability radius

                    else:
                        if dist > self.args.obs_radius[rover_type_ref]: continue #Observability radius

                else:
                    if dist > self.args.obs_radius[rover_type_ref]: continue  # Observability radius

                #if dist > self.args.obs_radius[rover_type_ref]: continue

                angle -= self_orient
                if angle < 0: angle += 360

                try: bracket = int(angle / self.args.angle_res)
                except: bracket = 0

                bracket = (bracket * self.args.num_poi_types) + poi_type

                if bracket >= len(temp_poi_dist_list):
                    #print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_poi_dist_list))
                    bracket = len(temp_poi_dist_list)-1
                #if dist == 0: dist = 0.001
                #if dist < 0.01: dist = 0.01
                if dist < 0.01: dist = 0.01
                temp_poi_dist_list[bracket].append((self.poi_value[poi_type][poi_id % (1 if poi_type == 0 else poi_type)]/(dist*dist))) # this is for a rover (as rover can see the POI within its observability region)

                #update closest POI for each rover info
                if dist < self.rover_closest_poi[rover_id][poi_type]: self.rover_closest_poi[rover_id][poi_type] = dist

                '''
                if dist < self.rover_closest_poi_2[rover_id]:
                    if rover_id not in self.poi_visitor_list[poi_id]:
                        self.rover_closest_poi_2[rover_id] = dist
                
                if dist > self.rover_farthest_poi[rover_id]: self.rover_farthest_poi[rover_id] = dist
                if dist > self.rover_farthest_poi_2[rover_id]:
                    if rover_id not in self.poi_visitor_list[poi_id]:
                        self.rover_farthest_poi_2[rover_id] = dist
                '''

            # Log all distance into brackets for other drones
            for id, loc, in enumerate(self.rover_pos):
                if id == rover_id: continue #Ignore self

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                #rover_type = int(id/self.args.num_uavs) # todo: getting type from rover ID
                rover_type = int(0) if id < self.args.num_uavs else int(1)
                angle -= self_orient
                if angle < 0: angle += 360

                #### consider only those rovers which are inside the observation radius
                ## todo: this is added for long_range_lidar of truck
                if self.args.config == 'fire_truck_uav_long_range_lidar':
                    try: bracket = int(angle / self.args.angle_res)
                    except: bracket = 0

                    if (rover_type_ref == 1) and bracket == 0: # as 0 is the longest range lidar in truck
                        if dist > self.args.long_range: continue  # Observability radius

                    else:
                        if dist > self.args.obs_radius[rover_type_ref]: continue #Observability radius of ref_type rover, as we are taking its range POIs and rovers

                else:
                    if dist > self.args.obs_radius[rover_type_ref]: continue #Observability radius of ref_type rover, as we are taking its range POIs and rovers


                #if dist > self.args.obs_radius[rover_type_ref]: continue

                #if dist == 0: dist = 0.001
                #if dist < 0.01: dist = 0.01
                if dist < 0.01: dist = 0.01
                try: bracket = int(angle / self.args.angle_res)

                #try: bracket = int(angle / (self.args.angle_res))

                except: bracket = 0

                bracket = (bracket * self.args.num_agent_types) + rover_type

                if bracket >= len(temp_rover_dist_list):
                    #print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_rover_dist_list), angle)
                    bracket = len(temp_rover_dist_list)-1
                temp_rover_dist_list[bracket].append((1/(dist*dist)))


                #update closest POI for each rover info #todo: added just for fire truck and UAV case

                ##### UAV keeps distance to closest trucks and trucks keeps distance to closest UAVs
                if (rover_type_ref == 0): # UAV
                    if (rover_type != rover_type_ref and dist < self.uav_closest_rover[rover_id]):
                        self.uav_closest_rover[rover_id] = dist
                    elif(rover_type == rover_type_ref and dist < self.uav_closest_uav[rover_id]):
                        self.uav_closest_uav[rover_id] = dist

                if (rover_type_ref == 1): # Truck
                    if (rover_type != rover_type_ref and  dist < self.rover_closest_uav[rover_id]):  # closest UAV to truck
                        self.rover_closest_uav[rover_id] = dist
                    elif(rover_type == rover_type_ref and dist < self.rover_closest_rover[rover_id]):
                        self.rover_closest_rover[rover_id] = dist

            ####Encode the information onto the state
            for bracket in range(len(temp_poi_dist_list)):
                # POIs of each type

                num_poi = len(temp_poi_dist_list[bracket])
                if num_poi > 0:
                    if self.args.sensor_model == 'density': poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi #Density Sensor
                    elif self.args.sensor_model == 'closest': poi_state[bracket] = max(temp_poi_dist_list[bracket])  #Closest Sensor
                    else: sys.exit('Incorrect sensor model')
                else: poi_state[bracket] = -1.0

                #Rovers
            for bracket in range(len(temp_rover_dist_list)):
                num_agents = len(temp_rover_dist_list[bracket])
                if num_agents > 0:
                    if self.args.sensor_model == 'density': rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents #Density Sensor
                    elif self.args.sensor_model == 'closest':
                        rover_state[bracket] = max(temp_rover_dist_list[bracket]) #Closest Sensor

                    else: sys.exit('Incorrect sensor model')
                else: rover_state[bracket] = -1.0 # -1 indicates no rover in that range, i.e lidar sensor is not getting reflected back


            state = rover_state + [rover_id] +  poi_state + self.rover_vel[rover_id] #Append rover_id, rover LIDAR and poi LIDAR to form the full state

            #print("%%%%%%%%%%%%%% LENGTH OF STATE: ", len(state))

            #if(len(state)!=111):
            #	print("here is the problem: ", len(state))
            # #Append wall info
            # state = state + [-1.0, -1.0, -1.0, -1.0]
            # if self_x <= self.args.obs_radius: state[-4] = self_x
            # if self.args.dim_x - self_x <= self.args.obs_radius: state[-3] = self.args.dim_x - self_x
            # if self_y <= self.args.obs_radius :state[-2] = self_y
            # if self.args.dim_y - self_y <= self.args.obs_radius: state[-1] = self.args.dim_y - self_y

            #state = np.array(state)
            joint_state.append(state)

            #if np.max(joint_state) > 100:
            #    print("Here is the problem")

        return joint_state


    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        v1 = x2 - x1;
        v2 = y2 - y1
        angle = np.rad2deg(np.arctan2(v1, v2))
        if angle < 0: angle += 360
        if math.isnan(angle): angle = 0.0

        dist = v1 * v1 + v2 * v2
        dist = math.sqrt(dist)

        if math.isnan(angle) or math.isinf(angle): angle = 0.0

        return angle, dist


    def get_local_reward(self): # different policy_choice for each agent
        #Update POI's visibility

        poi_visitors = [[[] for _ in range(self.args.num_agent_types)] for _ in range(self.args.num_poi)]
        poi_visitor_dist = [[[] for _ in range(self.args.num_agent_types)] for _ in range(self.args.num_poi)]

        # poi_visitors = np.zeros((self.args.num_poi, self.args.num_agent_types)) # fixme: changed [[] for _ in range(self.args.num_poi)]
        # poi_visitor_dist = np.zeros((self.args.num_poi, self.args.num_agent_types))

        for i, loc in enumerate(self.poi_pos): #For all POIs
            #if self.poi_status[i]== 0 and self.reward_type_idx== -1: # if it has been observed # removing condition on local rewards

            # POI type
            poi_type = 0
            if self.args.num_poi_A <= i < self.args.num_poi_A + self.args.num_poi_B: poi_type = 1
            if self.args.num_poi_A + self.args.num_poi_B <= i < self.args.num_poi: poi_type = 2

            if self.poi_status[poi_type][i if poi_type == 0 else i % self.args.num_poi_A] == 0:  # if it has been observed, ignore that POI
                continue #Ignore POIs that have been harvested already, making local reward independent on global reward

            for rover_id in range(self.args.num_agents): #For each rover (num_agents is the total number of agents)
                #agent_type = rover_id // self.args.num_uavs # for each of the rover, finding its type
                agent_type = int(0) if rover_id < self.args.num_uavs else int(1)
                x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
                dist = math.sqrt(x1 * x1 + y1 * y1) # distance of agent from POI
                if dist <= self.args.act_dist: # if distance is less, consider it as observed and capture the corresponding rover ID and distance
                    poi_visitors[i][agent_type].append(rover_id) # need to preserve the rover's ID so as to alot the rewards to them
                    #poi_visitors[i,agent_type] += 1 # Add rover to POI's visitor list of a particular agent type
                    poi_visitor_dist[i][agent_type].append(dist) #Add distance to POI's visitor list of a particular agent type

        #Compute reward
        rewards = [[0.0 for _ in range(len(self.args.reward_types))] for _ in range(self.args.num_agents)]

        for poi_id, rovers in enumerate(poi_visitors): # here rovers will be an list with all the IDs of types of agents (so 1 list for UAVs and 1 for rovers) rovers
                #if self.task_type == 'rover_tight' and len(rovers) >= self.args.coupling or self.task_type == 'rover_loose' and len(rovers) >= 1:
                #Update POI status

                #if self.task_type == 'rover_tight' and (len(rovers[i]) >= self.args.coupling for i in range(len(rovers))) or self.task_type == 'rover_heterogeneous' and (len(rovers[m]) + len(rovers[m+1]) >= 1 for m in range(len(rovers)-1)) or self.task_type == 'rover_loose' and (len(rovers[i]) + len(rovers[i+1]) >= 1 for i in range(len(rovers)-1)) or self.task_type == 'rover_trap' and (len(rovers[i]) >= 1 for i in range(len(rovers))):

                #if self.task_type == 'rover_heterogeneous' and sum(len(rovers[m]) for m in range(len(rovers))) >= 1: # todo: this case if the heterogeneity does not matter (one rover of any type is enough)
                poi_type = 0
                if self.args.num_poi_A <= poi_id < self.args.num_poi_A + self.args.num_poi_B: poi_type = 1
                if self.args.num_poi_A + self.args.num_poi_B <= poi_id < self.args.num_poi: poi_type = 2

                ## Todo: changed this
                temp_list = [item for sublist in rovers for item in sublist] # coverting list of list to a list
                self.poi_visitor_list[poi_id] = list(set(self.poi_visitor_list[poi_id]+temp_list)) # add the number corresponding to each agent

                for m in range(len(rovers)):
                    if (len(rovers[m]) >= self.coupling[m]):
                        coupling_satisfied = True
                        continue
                    else:
                        coupling_satisfied = False
                        break

                if self.task_type == 'rover_heterogeneous' and coupling_satisfied: # this is coupling of current POI
                    if (poi_type == 0):
                        self.poi_status[poi_type][poi_id] -= 1 # if coupling is fulfilled, make it 0 from 1
                    else:
                        for k in range(0, poi_type):
                            prev_observed = False
                            if sum(self.poi_status[k]) < len(self.poi_status[k]) :# even if 1 POI of previous type is observed
                                prev_observed = True

                        if prev_observed:
                            self.poi_status[poi_type][poi_id if poi_type == 0 else poi_id % self.args.num_poi_A] -= 1  # if coupling is fulfilled, make it 0 from 1


        ################################################################################
        ############### Proximity Rewards for different reward types ###################
        ################################################################################

        if self.args.is_proxim_rew:
            for i in range(self.args.num_agents):
                #print("%%%%%%%%%", self.rover_closest_poi)

                #agent_type = int(i / self.args.num_uavs)
                agent_type = int(0) if i < self.args.num_uavs else int(1)

                ###### sending all types of local rewards all at a time ########
                agent_proxim_reward = []

                ##### reward_types:
                #'go_to_poi'
                #'go_to_uav'
                #'go_to_truck'

                if (agent_type == 0):  # for UAV
                    # go to nearest POI of each type
                    for poi in range(self.args.num_poi_types):
                        ## go to nearest poi
                        if (self.rover_closest_poi[i][poi] <= self.args.obs_radius[agent_type]):  # POI is outside the range of truck
                            proxim_rew = self.args.act_dist / self.rover_closest_poi[i][poi]
                        else:  proxim_rew = 0.0 # reward will be in according to distance from POI as well as distance to UAV

                        if proxim_rew > 1.0: proxim_rew = 1.0
                        agent_proxim_reward.append(proxim_rew)
                        agent_proxim_reward.append(-proxim_rew)

                    ## go to uav
                    if (self.uav_closest_uav[i] <= self.args.obs_radius[agent_type]):  # for UAV to be close to truck
                        proxim_rew = self.args.act_dist / self.uav_closest_uav[i]
                    else:
                        proxim_rew = 0.0
                    agent_proxim_reward.append(proxim_rew)
                    agent_proxim_reward.append(-proxim_rew)


                else: # Truck
                    ## go to poi
                    if (self.rover_closest_poi[i][poi]  <= self.args.obs_radius[agent_type]):  # if the closest distance to POI is outside the obs radius, do nothing
                        proxim_rew = self.args.act_dist / self.rover_closest_poi[i][poi]   # both UAVs and fire trucks needs to go to POI, although the coupling requirement does not need UAV to go to them
                    else: proxim_rew = 0.0

                    if proxim_rew > 1.0: proxim_rew = 1.0
                    agent_proxim_reward.append(proxim_rew)

                    if self.args.reward_scheme == 'multiple':
                        ## go to uav
                        if (self.rover_closest_uav[i] <= self.args.obs_radius[agent_type]):
                            proxim_rew = self.args.act_dist / self.rover_closest_uav[i]
                        else: proxim_rew = 0.0

                        if proxim_rew > 1.0: proxim_rew = 1.0
                        agent_proxim_reward.append(proxim_rew)

                        ## go to truck
                        if (self.rover_closest_rover[i] <= self.args.obs_radius[agent_type]):  # if both UAV and other truck is outside the range of truck
                            proxim_rew = self.args.act_dist / self.rover_closest_rover[i]
                        else:
                            proxim_rew = 0.0

                        if proxim_rew > 1.0: proxim_rew = 1.0
                        agent_proxim_reward.append(proxim_rew)

                #rewards[i] += agent_proxim_reward
                rewards[i] = [a  + b for a, b in zip(rewards[i], agent_proxim_reward)]

                #print(self.rover_closest_poi[i], proxim_rew)
                #self.cumulative_local[i] += agent_proxim_reward
                self.cumulative_local[i]  = [a + b for a, b in zip(self.cumulative_local[i], agent_proxim_reward)]

        # Reset all distances
        self.rover_closest_poi = {}
        for i in range(self.args.num_agents):
            self.rover_closest_poi[i] = [self.args.dim_x * 2 for _ in range(self.args.num_poi_types)]  # distance of all rovers (trucks+uavs) from POIs

        self.uav_closest_rover = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]
        self.rover_closest_uav = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]  # distance of trucks from UAVs
        self.uav_closest_uav = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]  # distance of all uavs from a UAV
        self.rover_closest_rover = [self.args.dim_x * 2 for _ in range(self.args.num_agents)]  # distance of fire trucks from other fire trucks

        #print("****", rewards)
        return rewards


    def dummy_transition(self):
        joint_state = [[0.0 for _ in range(int(360 * (1+self.args.num_agent_types)/ self.args.angle_res)+3)] for _ in range(self.args.num_agents)]
        #rewards = [0.0 for _ in range(self.args.num_agents)]
        rewards = [[0.0 for _ in range(len(self.args.reward_types))] for _ in range(self.args.num_agents)]

        return joint_state, rewards, True, None


    def get_global_reward(self):
        global_rew = 0.0; max_reward = 0.0

        if self.task_type == 'rover_tight' or self.task_type == 'rover_loose' or self.task_type == 'rover_heterogeneous':
            poi_id = -1
            for i in range(self.args.num_poi_types):

                poi_id += 1
                # POI type
                poi_type = 0
                if self.args.num_poi_A <= poi_id < self.args.num_poi_A + self.args.num_poi_B: poi_type = 1
                if self.args.num_poi_A + self.args.num_poi_B <= poi_id < self.args.num_poi:
                    poi_type = 2

                for value, status in zip(self.poi_value[i], self.poi_status[i]):
                    global_rew += (status == 0) * value # values of POIs that have been observed
                    max_reward += value # all POIs

        # todo: change this
        elif self.task_type == 'rover_trap':  # Rover_Trap domain
            for value, visitors in zip(self.poi_value, self.poi_visitor_list):
                multiplier = len(visitors) if len(visitors) < self.coupling else self.coupling
                global_rew += value * multiplier
                max_reward += self.coupling * value

        else:
            sys.exit('Incorrect task type')

        global_rew = global_rew/max_reward

        return global_rew


    def render(self):
        # Visualize
        grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]
        symbols_agents = ['U', 'T'] # for UAV and truck
        symbol_pois = ['#_A', '#_B', '#_C']
        symbol_observed_pois = ['$_A', '$_B', '$_C']

        # Draw in rover path
        for rover_id, path in enumerate(self.rover_path):
            count= 0
            rover_type = int(0) if rover_id < self.args.num_uavs else int(1)

            for loc in path:
                count = count + 1
                x = int(loc[0]); y = int(loc[1])
                #grid[x][y] = str(rover_id)
                #print("$$$$$$$ X, Y COORDINATES $$$$$$$$", rover_id, x,y)

                #if (rover_id == 0): # only for UAVs
                if x < self.args.dim_x and y < self.args.dim_y and x >=0 and y >=0:
                    grid[x][y] = str(symbols_agents[rover_type]) + str(rover_id) + str("_") + str(count) # it will give exact how its travelling
                #else:
                #    print(str(rover_id) + str("_") + str(count),"---- WENT OUTSIDE TO ", (x, y))

            # Draw in food
        for poi_id in range(self.args.num_poi):

            poi_type = 0
            if self.args.num_poi_A <= poi_id < self.args.num_poi_A + self.args.num_poi_B: poi_type = 1
            if self.args.num_poi_A + self.args.num_poi_B <= poi_id < self.args.num_poi: poi_type = 2

            x = int(self.poi_pos[poi_id][0]);
            y = int(self.poi_pos[poi_id][1])
            marker = symbol_observed_pois[poi_type] if self.poi_status[poi_type][poi_id if poi_type == 0 else poi_id % self.args.num_poi_A] == 0 else symbol_pois[poi_type]
            grid[x][y] = marker

        for row in grid:
            print(row)

        for agent_id, temp in enumerate(self.action_seq):
            print()
            #print('Action Sequence Rover ', str(agent_id),)
            #for entry in temp:
            #    print(['{0: 1.1f}'.format(x) for x in entry], end =" " )
        print()

        print('------------------------------------------------------------------------')


    def viz(self, save=False, fname=''):

        padding = 70

        observed = 3 + self.args.num_agents*2
        unobserved = observed + 3

        # Empty Canvas
        matrix = np.zeros((padding*2+self.args.dim_x*10, padding*2+self.args.dim_y*10))


        #Draw in rover
        color = 10.0; rover_width = 1; rover_start_width = 4
        # Draw in rover path
        for rover_id, path in enumerate(self.rover_path):
            start_x, start_y = int(path[0][0]*10)+padding, int(path[0][1]*10)+padding
            color += 5

            matrix[start_x-rover_start_width:start_x+rover_start_width, start_y-rover_start_width:start_y+rover_start_width] = color
            #continue
            for loc in path[1:]:
                x = int(loc[0]*10)+padding; y = int(loc[1]*10)+padding
                if x > len(matrix) or y > len(matrix) or x < 0 or y < 0 : continue

                #Interpolate and Draw
                for i in range(int(abs(start_x-x))):
                    if start_x > x:
                        matrix[x+i - rover_width:x+i + rover_width, start_y - rover_width:start_y + rover_width] = color
                    else:
                        matrix[x - i - rover_width:x - i + rover_width, start_y - rover_width:start_y + rover_width] = color

                for i in range(int(abs(start_y-y))):
                    if start_y > y:
                        matrix[x - rover_width:x + rover_width, y + i- rover_width:y +i+ rover_width] = color
                    else:
                        matrix[x  - rover_width:x + rover_width, y-i- rover_width:y-i + rover_width] = color
                start_x, start_y = x, y



        #Draw in POIs
        poi_width = 8
        color += 10
        for poi_pos, poi_status in zip(self.poi_pos, self.poi_status):
            x = padding+int(poi_pos[0])*10; y = padding+int(poi_pos[1])*10
            if poi_status: color = unobserved #Not observed
            else: color = observed #Observed

            matrix[x-poi_width:x+poi_width, y-poi_width:y+poi_width] = color

        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='Accent', origin='upper')
        if save:
            plt.savefig(fname=fname, dpi=300, quality=90, format='png')
        else:
            plt.show()
