from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from ddpg import DDPGAgent
from memory import SequentialMemory
from rand import OrnsteinUhlenbeckProcess

#import Env as enviornment
from StormwaterEnvNew import StormwaterEnv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
This script runs a ddpg algorithm for control
of stormwater systems using a SWMM model as the environment

Author: Sami Saliba
Date: July 17, 2019

"""

env = StormwaterEnv()
nb_actions = 3

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + (11,)))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(8))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (11,), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=10,
                  random_process=random_process, gamma=.995, target_model_update=1e-3)

agent.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])
agent.fit(env, nb_steps=10000, visualize=False, verbose=0, nb_max_episode_steps=95)   
#agent.save_weights('weights/ddpg_{}_weights.h5f'.format("stormwater"), overwrite=True)
agent.test(env, nb_episodes=15, visualize=False, nb_max_episode_steps=95, plt="") 