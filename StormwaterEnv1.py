import numpy as np
import random
from reward_functions import reward_function2 as reward_function
from pyswmm import Simulation, Nodes, Links
import matplotlib.pyplot as plt
import gym


class StormwaterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, R=6, J=4, swmm_inp="data/simple3.inp"):


        super(StormwaterEnv, self).__init__()
        self.total_rewards = []
        self.eps_reward = 0

        self.num_steps = 0

        self.timestep = 0

        self.swmm_inp = swmm_inp        
        self.temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
        self.temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting
       
        self.R = R
        self.J = J
        # self.St1_depth = []
        # self.St2_depth = []
        # self.J3_depth = []
        # self.St1_flooding = []
        # self.St2_flooding = []
        # self.J3_flooding = []
        # self.R1_position = []
        # self.R2_position = []
            
        self.log_dumps = []

    def reset(self):
        self.swmm_inp="data/simple3.inp"
        
        self.total_rewards.append(self.eps_reward)
        #self.swmm_inp = 'simple%s.inp'%(random.randint(1,3))

        self.timestep += 1
        self.eps_reward = 0

        self.num_steps = 0
        
        self.done = False

        self.current_step = 0


        self.sim = Simulation(self.swmm_inp)

        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)

        self.sim.start()

        # print(self.sim.end_time, self.sim.start_time)
        self.sim_len = self.sim.end_time - self.sim.start_time
        self.num_steps = int(self.sim_len.total_seconds()/self.control_time_step)
        # print(self.num_steps)
        for i in range(1, self.R+1):
            self.link_object['R'+str(i)].target_setting = random.randrange(1)

        self.settings = [self.link_object['R' + str(i)].current_setting for i in range(1, self.R+1)]

        # one for loop would be faster but less readable
        # making new lists all the time is probably bad
        self.depths = [self.node_object['St' + str(i)].depth for i in range(1, self.R+1)]
        self.depths.extend([self.node_object['J'+str(i)].depth for i in range(2, self.J+1)])
       
        self.flooding = [self.node_object['St' + str(i)].flooding for i in range(1, self.R+1)]
        self.flooding.extend([self.node_object['J'+str(i)].flooding for i in range(2, self.J+1)])
     
        self.reward = reward_function(self.depths, self.flooding)
        
        self.eps_reward += self.reward
        
        state = []
        
        state = self.settings.copy()
        state.extend(self.depths)
        state.extend(self.flooding)
            
        return state


    def step(self, action):
        self.current_step += 1 # Decide whether this should be before or after taking actions

        for i in range(1, self.R+1):
            self.link_object['R'+str(i)].target_setting = action[i-1]

        self.sim.__next__()
        
        self.settings = [self.link_object['R' + str(i)].current_setting for i in range(1, self.R+1)]

        # one for loop would be faster but less readable
        # making new lists all the time is probably bad
        self.depths = [self.node_object['St' + str(i)].depth for i in range(1, self.R+1)]
        self.depths.extend([self.node_object['J'+str(i)].depth for i in range(2, self.J+1)])
       
        self.flooding = [self.node_object['St' + str(i)].flooding for i in range(1, self.R+1)]
        self.flooding.extend([self.node_object['J'+str(i)].flooding for i in range(2, self.J+1)])
       
        
        self.reward = reward_function(self.depths, self.flooding)

        self.eps_reward += self.reward

        state = []

        state = self.settings.copy()
        state.extend(self.depths)
        state.extend(self.flooding)


        if self.current_step == 1:
            self.log_dumps.append([])
        self.log_dumps[-1].append((self.reward, [self.settings, self.depths, self.flooding], self.timestep))
        return state, self.reward, self.done #, {} # {} is debugging information
    
    def close(self):
        self.sim.report()
        self.sim.close()

    def render(self, mode="human"):
        # This's probably not useful at all at the moment but should be here

       # return str(self.St1_depth) + "\n" + str(self.St2_depth) + "\n" + str(self.J3_depth) + "\n" + \
        #str(self.St1_flooding) + "\n" + str(self.St2_flooding) + "\n" + str(self.J3_flooding) + "\n" + \
        #str(self.R1_position) + "\n" + str(self.R2_position) + "\n"
        print("sls")

    def graph(self, location):
        for plot, dump in enumerate(self.log_dumps[-1:]):
            settings, depths, floodings = [[] for i in range(self.R)], [[] for i in range(self.R + self.J - 1)], [[] for i in range(self.R + self.J - 1)]
            rewards = []
        
            for reward, state, timestep in dump:

                rewards.append(reward)
                setting, depth, flooding = state
                
                for i, R in enumerate(settings):
                    R.append(setting[i])
                for i, S_depth in enumerate(depths):
                    S_depth.append(depth[i])
                for i, S_flooding in enumerate(floodings):
                    S_flooding.append(flooding[i])
            

            for i in range(1, self.R+1):
                plt.subplot(2, 3, i)
                plt.plot(settings[i-1])
                plt.ylim(0, 1)
                plt.title('R' + str(i))
                plt.ylabel("Valve Opening")
                plt.xlabel("time step")
            
            plt.tight_layout()
            if(plot == 5):
                plt.savefig(location + "TEST_" + str(timestep) + "_STATES", dpi=300)
            else:
                plt.savefig(location + str(timestep) + "_STATES", dpi=300)
           
            plt.clf()
            
            for i in range(1, self.R+1):
                plt.subplot(2, 3, i)
                plt.plot(depths[i-1])
                plt.ylim(0, 5)
                plt.title('St' + str(i) + " Depth")
                plt.ylabel("Feet")
                plt.xlabel("time step")


            plt.tight_layout()
            if(plot == 5):
                plt.savefig(location + "TEST_" + str(timestep) + "_DEPTHS", dpi=300)
            else:
                plt.savefig(location + str(timestep) + "_DEPTHS", dpi=300)
           
            plt.clf()

            for i in range(2, self.J+1):
                plt.subplot(2, 2, i-1)
                plt.plot(floodings[self.R + i-2])
                plt.title('J' + str(i) + " flooding")
                plt.ylabel("Feet")
                plt.xlabel("time step")
            

            plt.subplot(2, 2, 4)
            plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8], [sum(floodings[i]) for i in range(len(floodings))], tick_label=["St1","St2","St3","St4","St5","St6", "J2", "J3", "J4"])
            plt.ylim(0)
            plt.title('total_flooding')
            plt.ylabel("10^3 cubic feet")
            plt.xlabel("time step")

            plt.tight_layout()

            if(plot == 5):
                plt.savefig(location + "TEST_" + str(timestep) + "_FLOODING", dpi=300)
            else:
                plt.savefig(location + str(timestep) + "_FLOODING", dpi=300)
            
            plt.clf() 


            plt.subplot(2, 1, 1)
            plt.plot(rewards)
            # plt.ylim(0, 5)
            plt.title('Rewards')
            plt.ylabel("Reward")
            plt.xlabel("time step")

            plt.subplot(2, 1, 2)
            plt.ylim(-5000,0)
            plt.plot(self.total_rewards)
            plt.title('Total Rewards')
            plt.ylabel("Reward")
            plt.xlabel("eps")

            plt.tight_layout()

            if(plot == 5):
                plt.savefig(location + "TEST_" + str(timestep) + "_REWARDS", dpi=300)
            else:
                plt.savefig(location + str(timestep) + "_REWARDS", dpi=300)
            
            plt.clf() 
