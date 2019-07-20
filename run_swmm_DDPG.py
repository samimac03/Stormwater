from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from ddpg import DDPGAgent
from memory import SequentialMemory
from rand import OrnsteinUhlenbeckProcess

#import Env as enviornment
from StormwaterEnv import StormwaterEnv

"""
This script runs Deep Q-Network RL algorithm for control
of stormwater systems using a SWMM model as the environment

Author: Sami Saliba
Date: July 17, 2019

"""



# Get the environment and extract the number of actions.
env = StormwaterEnv()
nb_actions = 2

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + (7,)))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + (7,), name='observation_input')
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


memory = SequentialMemory(limit=10000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=5, visualize=False, verbose=0, nb_max_episode_steps=95)

agent.save_weights('ddpg_{}_weights.h5f'.format("stormwater"), overwrite=True)

agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=95)


