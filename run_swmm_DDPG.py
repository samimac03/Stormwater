import os
import numpy as np
import matplotlib.pyplot as plt
from pyswmm import Simulation, Nodes, Links
from actor_critic import Actor, Critic
from replay_memory import ReplayMemoryAgent, random_indx, create_minibatch
from pyswmm_utils import OrnsteinUhlenbeckProcess, save_state, save_action, gen_noise
import Env

"""
This script runs Deep Q-Network RL algorithm for control
of stormwater systems using a SWMM model as the environment

Author: Sami Saliba
Date: July 17, 2019

"""
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess



# Get the environment and extract the number of actions.
env = new Env()
nb_actions = 2

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + 2))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + 2, name='observation_input')
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
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=10, visualize=False, verbose=0, nb_max_episode_steps=900)

agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)




plt.subplot(2, 2, 1)
plt.plot(St1_depth)
plt.ylim(0, 5)
plt.title('St1_depth')
plt.ylabel("ft")
plt.xlabel("time step")

plt.subplot(2, 2, 2)
plt.plot(St2_depth)
plt.ylim(0, 5)
plt.title('St2_depth')
plt.ylabel("ft")
plt.xlabel("time step")

plt.subplot(2, 2, 3)
plt.plot(J3_depth)
plt.ylim(0, 2)
plt.title('J3_depth')
plt.ylabel("ft")
plt.xlabel("time step")

# bar graph for total flooding
plt.subplot(2, 2, 4)
plt.bar([0, 1, 2], [sum(St1_flooding), sum(St2_flooding), sum(J3_flooding)], tick_label=["St1", "St2", "J3"])
plt.ylim(0)
plt.title('total_flooding')
plt.ylabel("10^3 cubic feet")

plt.tight_layout()
# plt.show()
plt.savefig("smart_stormwater_rl/RL_DDPG/plots/ddpg_model_results_" + str(best_episode) + rwd + ".png", dpi=300)
plt.close()

# plot rewards and actions
plt.subplot(2, 1, 1)
plt.plot(rewards_episode_tracker)
plt.ylabel("average reward")
plt.xlabel("episode")

plt.subplot(2, 1, 2)
plt.plot(R1_position)
plt.plot(R2_position, linestyle='--')
plt.ylim(0, 1)
plt.ylabel("orifice position")
plt.xlabel("time step")
plt.tight_layout()
plt.savefig("smart_stormwater_rl/RL_DDPG/plots/ddpg_model_rewards_" + str(num_episodes) + rwd + "epi" +
            str(best_episode) + ".png", dpi=300)
plt.close()
