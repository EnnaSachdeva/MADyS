

# Code accompanying "Multiagent Learning via Dynamic Skill Selection"

This code is written using the following packages:

- Python 3.6.9
- Pytorch 1.2
- Numpy 1.18.1
- Tensorboard

## Code labels 

train.py: Evolutionary learner that generates data which is stored in a shared Replay Buffer. This data is bootstrapped by policy gradient learners to learn low level skills. 

core/runner.py: Rollout worker

core/neuroevolution.py: Implements Sub-Structured Based Neuroevolution (SSNE) with a dynamic population

core/off_policy_algo.py: Implements the off_policy_gradient learner (TD3/DDPG) 

core/buffer.py: Cyclic Shared Replay buffer

core/models.py: Neural Network models for Neuroevolution and TD3/DDPG actors and critics

core/agent.py: Policy Selector agent, and primitive agents 

core/mod_utils.py: Helper functions

core/test.py: Test the trained networks

envs/rover_domain: multiagent coordination rover domain

envs/env_wrapper.py: wrapper for env

## To Run

python train.py --num_uav $NUM_UAV$ --num_poi_A $NUM_POI_A$ --num_poi_B $NUM_POI_B$ --num_poi_C $NUM_POI_C$ --num_poi_D $NUM_POI_D$ --poi_sequence $DESIRED TEMPORAL COUPLING, for instance- {0: None, 1: [0]}$ --coupling_uav $DESIRED SPATIAL COUPLING$ --seed $SEED$
 
