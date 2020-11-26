import numpy as np
import math
import random

class SequentialPOIRoverDomain:
    """
    Extends the rover domain class to include functions for sequential POI observation tasks,
    where the reward is not handed out until all types of a POI are observed, and there is a specific
    sequential relationship for how each POI can be observed.
    """

    def __init__(self, num_rovers, num_poi, num_steps, setup_size, poi_types, poi_sequence, num_reward_types, coupling_factor, id, **kwargs):
        """
        Sets up the Sequential POI observation world
        :param num_rovers: number of rovers (passed to parent)
        :param num_steps: number of timesteps in the world (passed to parent)
        :param num_poi: number of POI (used internally, and passed to parent)
        :param poi_types: List, the type of each poi in the world. Must specify a type for all POI
        :param poi_sequence: Dict, Graph sequence dictionary showing which parents each type has in the overall problem.
        :param kwargs: Extra kwargs (passed to parent)
        """

        """
        Sets up the rover domain
        :param number_rovers: number of rovers
        :param number_poi: number of POI
        :param n_steps: number of timesteps in the world
        :param n_req: number of agents to make an observation
        :param min_dist: minimum distance between the agent and a POI to "count" for observing
        :return:
        """

        self.n_rovers = num_rovers
        self.n_pois = num_poi
        self.poi_types = poi_types
        self.num_reward_types = num_reward_types
        random.seed(id)

        if len(poi_types) != num_poi:
            raise ValueError("Must specify POI type for ALL POI (length does not match num_poi)")

        # store the sequence as a graph, with each type pointing to the immediate parent(s) which must be satisfied
        # for it to be counted as a score.
        self.sequence = poi_sequence

        self.n_steps = num_steps

        self.n_req = coupling_factor
        self.min_dist = 1.0
        self.step_id = 0
        self.act_dist = 3
        # Done is set to true to prevent stepping the sim forward
        # before a call to reset (i.e. the domain is not yet ready/initialized).

        self.setup_size = setup_size
        self.interaction_dist = 1 # todo: need to change this.
        self.n_obs_sections = 4
        self.reorients = True
        self.discounts_eval = False

        # POI visited is a dictionary which keeps track of which types have been visited
        self.poi_visited = {}
        for t in self.sequence:
            self.poi_visited[t] = False

        self.poi_visitors = {} # this is for each index
        for poi in self.poi_types:
            self.poi_visitors[poi] = []

        # Override the size of the observations matrix
        self.rover_observations = np.zeros((self.n_rovers, 1 + len(self.poi_visited), self.n_obs_sections))

        # If user has not specified initial data, the domain provides
        # automatic initialization
        self.init_rover_positions = (np.random.rand(self.n_rovers, 2) * self.setup_size*3/4)
        rand_angles = np.random.uniform(-np.pi, np.pi, self.n_rovers)
        self.init_rover_orientations = np.vstack((np.cos(rand_angles), np.sin(rand_angles))).T

        self.poi_values = np.arange(self.n_pois) + 1.
        self.poi_positions = (np.random.rand(self.n_pois, 2) * self.setup_size)

        # If initial data is invalid (e.g. number of initial rovers does not
        # match init_rover_positions.shape[0]), we need to raise an error
        '''
        if self.init_rover_positions.shape[0] != self.n_rovers:
            raise ValueError('Number of rovers does not match number of initial ' + 'rover positions')
        if self.init_rover_orientations.shape[0] != self.n_rovers:
            raise ValueError('Number of rovers does not match number of initial ' + 'rover orientations')
        if self.poi_values.shape[0] != self.n_pois:
            raise ValueError('Number of POIs does not match number of POI values')
        if self.poi_positions.shape[0] != self.n_pois:
            raise ValueError('Number of POIs does not match number of POI positions')
        '''

        # Allocate space for all unspecified working data
        self.rover_positions = np.zeros((self.n_rovers, 2))
        self.rover_orientations = np.zeros((self.n_rovers, 2))
        self.rover_position_histories = np.zeros((self.n_steps + 1, self.n_rovers, 2))

        # Allocate space for all unspecified return data
        #self.rover_observations = np.zeros((self.n_rovers, 2, self.n_obs_sections))
        self.rover_rewards = np.zeros(self.n_rovers)

        '''
        # Reallocate all invalid working data arrays
        if self.rover_positions.shape[0] != self.n_rovers:
            self.rover_positions = np.zeros((self.n_rovers, 2))
        if self.rover_orientations.shape[0] != self.n_rovers:
            self.rover_orientations = np.zeros((self.n_rovers, 2))
        if (self.rover_position_histories.shape[0] != self.n_steps + 1 or self.rover_position_histories.shape[
            1] != self.n_rovers):
            self.rover_position_histories = np.zeros((self.n_steps + 1, self.n_rovers, 2))

        # Recreate all invalid return data
        if (self.rover_observations.shape[0] != self.n_rovers or self.rover_observations.shape[
            2] != self.n_obs_sections):
            self.rover_observations = np.zeros((self.n_rovers, 2, self.n_obs_sections))
        if self.rover_rewards.shape[0] != self.n_rovers:
            self.rover_rewards = np.zeros(self.n_rovers)
        '''

        # Copy over initial data to working data
        self.rover_positions[...] = self.init_rover_positions
        self.rover_orientations[...] = self.init_rover_orientations

        self.local_rewards = np.zeros((self.num_reward_types, self.n_rovers))

        self.rover_closest_poi= 100000*np.ones((self.n_rovers, len(set(self.poi_types))))
        self.rover_closest_rover= 100000 * np.ones(self.n_rovers)

        self.action_sequence = np.zeros((self.n_rovers, self.n_steps, 2))
        self.step_id = 0
        self.done = False


    def reset(self):
        """
        Resets the "poi_visited" dictionary to entirely false, in addition to the parent reset behavior.
        :return: None
        """
        self.step_id = 0
        self.done = False

        '''
        # Reallocate all invalid working data arrays
        if self.rover_positions.shape[0] != self.n_rovers:
            self.rover_positions = np.zeros((self.n_rovers, 2))
        if self.rover_orientations.shape[0] != self.n_rovers:
            self.rover_orientations = np.zeros((self.n_rovers, 2))
        if (self.rover_position_histories.shape[0] != self.n_steps + 1 or self.rover_position_histories.shape[1] != self.n_rovers):
            self.rover_position_histories = np.zeros((self.n_steps + 1, self.n_rovers, 2))

        # Recreate all invalid return data
        if (self.rover_observations.shape[0] != self.n_rovers or self.rover_observations.shape[2] != self.n_obs_sections):
            self.rover_observations = np.zeros((self.n_rovers, 2, self.n_obs_sections))
        if self.rover_rewards.shape[0] != self.n_rovers:
            self.rover_rewards = np.zeros(self.n_rovers)
        '''


        # Copy over initial data to working data
        self.init_rover_positions = (np.random.rand(self.n_rovers, 2) * self.setup_size * 3 / 4)
        self.rover_positions[...] = self.init_rover_positions
        self.rover_orientations[...] = self.init_rover_orientations

        # Store first rover positions in histories
        # todo avoiding slicing for speed?
        self.rover_position_histories[0, ...] = self.init_rover_positions

        self.local_rewards = np.zeros((self.num_reward_types, self.n_rovers))
        self.rover_closest_poi= 100000*np.ones((self.n_rovers, len(set(self.poi_types))))
        self.rover_closest_rover= 100000 * np.ones(self.n_rovers)
        self.action_sequence = np.zeros((self.n_rovers, self.n_steps, 2))

        for t in self.poi_types:
            self.poi_visited[t] = False

        self.poi_visitors = {} # this is for each index
        for poi in self.poi_types:
            self.poi_visitors[poi] = []

        self.update_observations()

        return self.rover_observations


    def step(self, actions, evaluate = None):
        """
        Provided for convenience, not recommended for performance
        """
        # TODO what on earth is the "evaluate"
        if not self.done:
            if actions is not None:
                self.move_rovers(actions)
            self.step_id += 1

            # We must record rover positions after increasing the step
            # index because the initial position before any movement
            # is stored in rover_position_histories[0], so the first step
            # (step_id = 0) must be stored in rover_position_histories[1]
            self.rover_position_histories[self.step_id, ...] = self.rover_positions

        self.done = self.step_id >= self.n_steps
        if evaluate:
            evaluate(self)
        else:
            self.update_rewards_step_global_eval()
        self.update_observations()
        
        self.update_sequence_visits()
        self.get_local_reward()

        return self.rover_observations, self.local_rewards, int(self.done),  self.sequential_score()


    def move_rovers(self, actions):

        # clip actions
        for rover_id in range(self.n_rovers):
            actions[rover_id, 0] = min(max(-1, actions[rover_id, 0]), 1) # clipping the actions between 0 and 1
            actions[rover_id, 1] = min(max(-1, actions[rover_id, 1]), 1)


        # Translate and Reorient all rovers based on their actions
        for rover_id in range(self.n_rovers):
            self.action_sequence[rover_id, self.step_id, 0] = actions[rover_id, 0]
            self.action_sequence[rover_id, self.step_id, 1] = actions[rover_id, 1]

            # turn action into global frame motion
            dx = (self.rover_orientations[rover_id, 0] * actions[rover_id, 0]
                  - self.rover_orientations[rover_id, 1] * actions[rover_id, 1])
            dy = (self.rover_orientations[rover_id, 0] * actions[rover_id, 1]
                  + self.rover_orientations[rover_id, 1] * actions[rover_id, 0])

            # globally move and reorient agent
            self.rover_positions[rover_id, 0] += dx
            self.rover_positions[rover_id, 1] += dy


            #if dx >= 1 or  dx <= -1 or dy >= 1 or  dy <= -1:
            #    print(dx, dy, "Bullshit")
            # Reorient agent in the direction of movement in the global
            # frame.  Avoid divide by 0 (by skipping the reorientation step
            # entirely).
            if not (dx == 0. and dy == 0.):
                norm = np.sqrt(dx * dx + dy * dy)
                self.rover_orientations[rover_id, 0] = dx / norm
                self.rover_orientations[rover_id, 1] = dy / norm


        '''
        if self.reorients:
            # Translate and Reorient all rovers based on their actions
            for rover_id in range(self.n_rovers):
                self.action_sequence[rover_id, self.step_id, 0] = actions[rover_id, 0]
                self.action_sequence[rover_id, self.step_id, 1] = actions[rover_id, 1]

                # turn action into global frame motion
                dx = (self.rover_orientations[rover_id, 0] * actions[rover_id, 0]
                      - self.rover_orientations[rover_id, 1] * actions[rover_id, 1])
                dy = (self.rover_orientations[rover_id, 0] * actions[rover_id, 1]
                      + self.rover_orientations[rover_id, 1] * actions[rover_id, 0])

                # globally move and reorient agent
                self.rover_positions[rover_id, 0] += dx
                self.rover_positions[rover_id, 1] += dy


                #if dx >= 1 or  dx <= -1 or dy >= 1 or  dy <= -1:
                #    print(dx, dy, "Bullshit")
                # Reorient agent in the direction of movement in the global
                # frame.  Avoid divide by 0 (by skipping the reorientation step
                # entirely).
                if not (dx == 0. and dy == 0.):
                    norm = np.sqrt(dx * dx + dy * dy)
                    self.rover_orientations[rover_id, 0] = dx / norm
                    self.rover_orientations[rover_id, 1] = dy / norm
        # Else domain translates but does not reorients agents
        else:
            for rover_id in range(self.n_rovers):
                self.rover_positions[rover_id, 0] += actions[rover_id, 0]
                self.rover_positions[rover_id, 1] += actions[rover_id, 1]

        '''
    def calc_step_eval_from_poi(self, poi_id):
        # todo profile the benefit (or loss) of TempArray

        sqr_dists_to_poi = [ 0 for _ in range(self.n_rovers)]

        # Get the rover square distances to POIs.
        for rover_id in range(self.n_rovers):
            displ_x = (self.rover_positions[rover_id, 0] - self.poi_positions[poi_id, 0])
            displ_y = (self.rover_positions[rover_id, 1] - self.poi_positions[poi_id, 1])
            sqr_dists_to_poi[rover_id] = displ_x * displ_x + displ_y * displ_y

        # Sort (n_req) closest rovers for evaluation
        # Sqr_dists_to_poi is no longer in rover order!

        #partial_sort(sqr_dists_to_poi.begin(),
        #             sqr_dists_to_poi.begin() + self.n_req,
        #             sqr_dists_to_poi.end())

        sqr_dists_to_poi.sort()

        # Is there (n_req) rovers observing? Only need to check the (n_req)th
        # closest rover
        #if sqr_dists_to_poi[self.n_req - 1] > self.interaction_dist * self.interaction_dist:
        #    return 0.

        for i in range(self.n_req):
            if sqr_dists_to_poi[i] > self.interaction_dist * self.interaction_dist:
                # Not close enough?, then there is no reward for this POI
                return 0.

        # Yes? Continue evaluation
        if self.discounts_eval:
            sqr_dist_sum = 0.
            # Get sum sqr distance of nearest rovers
            for near_rover_id in range(self.n_req):
                sqr_dist_sum += sqr_dists_to_poi[near_rover_id]
            return self.poi_values[poi_id] / max(self.min_dist, sqr_dist_sum)
        # Do not discount POI evaluation
        else:
            return self.poi_values[poi_id]



    def stop_prematurely(self):
        self.n_steps = self.step_id
        self.done = True



    def mark_POI_observed(self, poi_type, poi_index):
        """
        Updates the self.poi_visited[poi_type] if the dependencies are properly met
        Only does the update for the type being queried.

        :param poi_type: String, the type identifier of the poi being observed
        :return: None
        """

        parents = self.sequence[poi_type]
        if parents is not None:
            observed = True
            for p in parents:
                if not self.poi_visited[p]:
                    observed = False
                    break
            if (len(self.poi_visitors[poi_index]) >= self.n_req) and observed:
                self.poi_visited[poi_type] = True

        else:
            # If no parents
            if len(self.poi_visitors[poi_index]) >= self.n_req:
                self.poi_visited[poi_type] = True
                #import pdb
                #print("Point visitors", self.poi_visitors)
                #pdb.set_trace()
            #else: self.poi_visited[poi_type] = False
            

        '''
        parents = self.sequence[poi_type]
        if parents is not None:
            observed = True
            for p in parents:
                if not self.poi_visited[p]:
                    observed = False
                    break
            if observed:
                self.poi_visited[poi_type] = True
        else:
            # If no parents
            self.poi_visited[poi_type] = True
        '''

    def update_sequence_visits(self):
        """
        Uses current rover positions to update observations.

        :return: None
        """
        # TODO implement tight coupling in this scenario (can use parent class functionality?)
        '''
        for rover_idx, rover in enumerate(self.rover_positions):
            for i, poi in enumerate(self.poi_positions):
                # POI is within observation distance
                # print("Distance to {} : {}".format(np.array(poi), np.linalg.norm(np.array(rover) - np.array(poi))))
                if np.linalg.norm(np.array(rover) - np.array(poi)) < self.interaction_dist:
                    self.poi_visitors[i].append(rover_idx)
                    self.mark_POI_observed(self.poi_types[i], i)
        '''
        for i, poi in enumerate(self.poi_positions):
            for rover_idx, rover in enumerate(self.rover_positions):
                # POI is within observation distance
                # print("Distance to {} : {}".format(np.array(poi), np.linalg.norm(np.array(rover) - np.array(poi))))
                if np.linalg.norm(np.array(rover) - np.array(poi)) < self.interaction_dist:
                    if rover_idx not in self.poi_visitors[i]:
                        self.poi_visitors[i].append(rover_idx)
                    # check if the POI has been observed
                    self.mark_POI_observed(self.poi_types[i], i)

    def easy_sequential_score(self):
        """
        The global reward based on how far into the POI observation graph you get
        :return: Global reward based on graph completion
        """
        return sum([1 for x in self.poi_visited.values() if x])


    def sequential_score(self):
        """
        Checks if all the POI in poi_visited have been observed. Returns a fixed score at the moment.

        :return: Global reward score for the agents based on the sequential observation task
        """
        # Update to see if the requirements are now set
        # (done until no more updates are found, to account for cascading updates)
        # Now with the updated graph, we check if all POI are observed
        if all(self.poi_visited.values()):
            return 1
        else:
            return 0


    def add_to_sensor(self, rover_id, type_id, other_x, other_y, val):
        # Get global (gf) frame displacement
        gf_displ_x = (other_x - self.rover_positions[rover_id, 0])
        gf_displ_y = (other_y - self.rover_positions[rover_id, 1])

        # Set displacement value used by sensor to global frame
        displ_x = gf_displ_x
        displ_y = gf_displ_y

        # /May/ reorient displacement for observations
        if self.reorients:
            # Get rover frame (rf) displacement
            rf_displ_x = (self.rover_orientations[rover_id, 0] * displ_x
                          + self.rover_orientations[rover_id, 1] * displ_y)
            rf_displ_y = (self.rover_orientations[rover_id, 0] * displ_y
                          - self.rover_orientations[rover_id, 1] * displ_x)
            # Set displacement value used by sensor to rover frame
            displ_x = rf_displ_x
            displ_y = rf_displ_y

        dist = math.sqrt(displ_x * displ_x + displ_y * displ_y)

        # By bounding distance value we
        # implicitly bound sensor values
        if dist < self.min_dist:
            dist = self.min_dist

        # Get arc tangent (angle) of displacement
        angle = math.atan2(displ_y, displ_x)

        #  Get intermediate Section Index by discretizing angle
        sec_id_temp = math.floor((angle + math.pi) / (2 * math.pi) * self.n_obs_sections)

        # Clip and convert to get Section id
        sec_id = min(max(0, sec_id_temp), self.n_obs_sections - 1)

        self.rover_observations[rover_id, type_id, sec_id] += val / (dist * dist)

        ### TODO: added for local rewards
        if type_id > 0: # for POI types, not rover types
        # check the distance of rover from each POI type
            if dist < self.rover_closest_poi[rover_id, type_id-1]: self.rover_closest_poi[rover_id, type_id-1] = dist

        else: # for other agent, type_id == 0
            # check the distance of rover from each POI type
            if dist < self.rover_closest_rover[rover_id]: self.rover_closest_rover[rover_id] = dist



    # Needs to observe the different types of poi, and also make the observed POI dictionary into the state as well
    def update_observations(self):
        """
        Updates the internally held observations of each agent.
        State is represented as:
        <agents, poi type A, poi type B, ... , poi type N, poi A observed, poi B observed, ... , poi N observed>
        Where "agents" and each "poi type" are split into the quadrants centered around the rover.
        "poi X observed" is a 0-1 flag on whether or not a POI of type A has been successfully observed.

        Accommodates the reorienting/fixed frame differences.

        :return: None
        """
        # Reset the observation matrix
        for rover_id in range(self.n_rovers):
            for type_id in range(1 + len(self.poi_visited)):
                for section_id in range(self.n_obs_sections):
                    self.rover_observations[rover_id, type_id, section_id] = 0.
                    #self.rover_observations[rover_id, type_id, section_id] = -1

        for rover_id in range(self.n_rovers):
            # Update rover type observations
            for other_rover_id in range(self.n_rovers):
                # agents do not sense self (ergo skip self comparison)
                if rover_id == other_rover_id:
                    continue
                self.add_to_sensor(
                    rover_id,
                    0,
                    self.rover_positions[other_rover_id, 0],
                    self.rover_positions[other_rover_id, 1],
                    1.
                )
            # Update POI type observations
            for poi_id in range(self.n_pois):
                poi_type = self.poi_types[poi_id]
                # The index in this call is the poi type +1 (to offset the rovers first)
                # and then is assigned to it's respective section
                self.add_to_sensor(
                    rover_id,
                    poi_type + 1,
                    self.poi_positions[poi_id, 0],
                    self.poi_positions[poi_id, 1],
                    1.
                )



    def score_single_type(self, poi_type):
        """
        Calculates the step score for a single type of POI, ignoring all other POI

        :param poi_type: Th e type of the point of interest being scored
        :return: score, the global score for observing the specified POI type
        """
        score = 0
        for poi_id in range(self.n_pois):
            if self.poi_types[poi_id] == poi_type:
                score += self.calc_step_eval_from_poi(poi_id)
        return score

    def score_g_easy(self):
        score = 0
        for poi_id in range(self.n_pois):
            score += self.calc_step_eval_from_poi(poi_id)
        return score



    def calc_step_global_eval(self):
        eval = 0.
        for poi_id in range(self.n_pois):
            eval += self.calc_step_eval_from_poi(poi_id)

        return eval



    def calc_step_cfact_global_eval(self, rover_id):
        # Hack: simulate counterfactual by moving agent FAR AWAY, then calculate
        far = 1000000.  # That's far enough, right?

        # Store actual positions for later reassignment
        actual_x = self.rover_positions[rover_id, 0]
        actual_y = self.rover_positions[rover_id, 1]

        # Move rover artificially
        self.rover_positions[rover_id, 0] = far
        self.rover_positions[rover_id, 1] = far

        # Calculate /counterfactual/ evaluation
        evaluation = self.calc_step_global_eval()

        # Move rover back
        self.rover_positions[rover_id, 0] = actual_x
        self.rover_positions[rover_id, 1] = actual_y

        return evaluation


    def calc_traj_global_eval(self):
        # Only evaluate trajectories at the end
        if not self.done:
            return 0.

        # Initialize evaluations to 0
        eval = 0.
        evaluation = 0
        poi_evals = [0 for _ in range(self.n_pois)]
        for poi_id in range(self.n_pois):
            poi_evals[poi_id] = 0

        # Get evaluation for poi, for each step, storing the max
        for step_id in range(self.n_steps + 1):
            # Go back in time
            self.rover_positions[...] = \
                self.rover_position_histories[step_id, ...]

            # Keep best step eval for each poi
            for poi_id in range(self.n_pois):
                poi_evals[poi_id] = max(poi_evals[poi_id],
                                        self.calc_step_eval_from_poi(poi_id))

        # Set evaluation to the sum of all POI-specific evaluations
        for poi_id in range(self.n_pois):
            evaluation += poi_evals[poi_id]

        return evaluation




    def calc_traj_cfact_global_eval(self, rover_id):
        # Hack: simulate counterfactual by moving agent FAR AWAY, then calculate
        far = 1000000.  # That's far enough, right?

        actual_x_hist = [0 for _ in range(self.n_steps + 1)]
        actual_y_hist = [0 for _ in range(self.n_steps + 1)]

        for step_id in range(self.n_steps + 1):
            # Store actual positions for later reassignment
            actual_x_hist[step_id] = \
                self.rover_position_histories[step_id, rover_id, 0]
            actual_y_hist[step_id] = \
                self.rover_position_histories[step_id, rover_id, 1]

            # Move rover artificially
            self.rover_position_histories[step_id, rover_id, 0] = far
            self.rover_position_histories[step_id, rover_id, 1] = far

        # Calculate /counterfactual/ evaluation
        evaluation = self.calc_traj_global_eval()

        for step_id in range(self.n_steps + 1):
            # Move rover back
            self.rover_position_histories[step_id, rover_id, 0] = \
                actual_x_hist[step_id]
            self.rover_position_histories[step_id, rover_id, 1] = \
                actual_y_hist[step_id]

        return evaluation



    def update_rewards_step_global_eval(self):
        global_eval = self.calc_step_global_eval()

        for rover_id in range(self.n_rovers):
            self.rover_rewards[rover_id] = global_eval

    def update_rewards_step_diff_eval(self):
        global_eval = self.calc_step_global_eval()
        for rover_id in range(self.n_rovers):
            cfact_global_eval = self.calc_step_cfact_global_eval(rover_id)
            self.rover_rewards[rover_id] = global_eval - cfact_global_eval

    def update_rewards_traj_global_eval(self):
        global_eval = self.calc_traj_global_eval()
        for rover_id in range(self.n_rovers):
            self.rover_rewards[rover_id] = global_eval

    def update_rewards_traj_diff_eval(self):
        global_eval = self.calc_traj_global_eval()
        for rover_id in range(self.n_rovers):
            cfact_global_eval = self.calc_traj_cfact_global_eval(rover_id)
            self.rover_rewards[rover_id] = global_eval - cfact_global_eval




    def get_local_reward(self):

        self.local_rewards = [[0.0 for _ in range(self.num_reward_types)] for _ in range(self.n_rovers)]

        for rover_id in range(self.n_rovers):
            rover_proxim_reward = []
            # local reward for each POI type
            for poi_type in range(len(set(self.poi_types))):
                if self.rover_closest_poi[rover_id, poi_type] <=  self.interaction_dist *  self.interaction_dist:
                    proxim_rew = self.act_dist / self.rover_closest_poi[rover_id, poi_type]
                else:
                    proxim_rew = 0

                if proxim_rew > 1.0: proxim_rew = 1.0
                rover_proxim_reward.append(proxim_rew) # go to POI of a particular type
                rover_proxim_reward.append(-proxim_rew) # go away from POI of a particular type

            # local reward for closest agent
            if self.n_rovers > 1: # if more than 1 rovers are present
                if (self.rover_closest_rover[rover_id] <= self.interaction_dist *  self.interaction_dist):  # if both UAV and other truck is outside the range of truck
                    proxim_rew = self.act_dist / self.rover_closest_rover[rover_id]
                else:
                    proxim_rew = 0
                if proxim_rew > 1.0: proxim_rew = 1.0
                rover_proxim_reward.append(proxim_rew)
                rover_proxim_reward.append(-proxim_rew)

            self.local_rewards[rover_id] = [a + b for a, b in zip(self.local_rewards[rover_id], rover_proxim_reward)]

        self.rover_closest_poi = 100000 * np.ones((self.n_rovers, len(set(self.poi_types))))

        self.rover_closest_rover = 100000 * np.ones(self.n_rovers)

        return self.local_rewards



    def render(self):
        # Visualize

        #print("POI positions: ", self.poi_positions)
        #print("Rover positions: ", self.rover_positions)

        grid = [['-' for _ in range(self.setup_size)] for _ in range(self.setup_size)]
        symbols_agents = ['U', 'T'] # for UAV and truck
        symbol_pois = ['#_A', '#_B', '#_C', '#_D']
        symbol_observed_pois = ['$_A', '$_B', '$_C', '$_D']

        # Draw in rover path

        rover_type = 0
        for rover_id in range(self.n_rovers):
            count = 0
            for step in range(self.n_steps):
                count = count + 1
                x = int(self.rover_position_histories[step, rover_id, 0]); y = int(self.rover_position_histories[step, rover_id, 1])
                #grid[x][y] = str(rover_id)
                #print("$$$$$$$ X, Y COORDINATES $$$$$$$$", rover_id, x,y)

                #if (rover_id == 0): # only for UAVs
                if x < self.setup_size and y < self.setup_size and x >=0 and y >=0:
                    grid[x][y] = str(symbols_agents[rover_type]) + str(rover_id) + str("_") + str(count) # it will give exact how its travelling
                #else:
                #    print(str(rover_id) + str("_") + str(count),"---- WENT OUTSIDE TO ", (x, y))

            # Draw in food
        for poi_id in range(self.n_pois):
            poi_type = self.poi_types[poi_id]

            x = int(self.poi_positions[poi_id, 0]);
            y = int(self.poi_positions[poi_id, 1]);
            marker = symbol_observed_pois[poi_type] if self.poi_visited[poi_type] == True else symbol_pois[poi_type]
            grid[x][y] = marker

        for row in grid:
            print(row)
        
        '''       
        for agent_id in range(self.n_rovers):
            print(self.action_sequence[agent_id, :, :])
        '''


        #    print()
            #print('Action Sequence Rover ', str(agent_id),)
            #for entry in temp:
            #    print(['{0: 1.1f}'.format(x) for x in entry], end =" " )
        print()

        print('------------------------------------------------------------------------')

