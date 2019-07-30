import numpy as np
import random
from reward_functions import reward_function2 as reward_function
from pyswmm import Simulation, Nodes, Links
import matplotlib.pyplot as plt
import gym


class StormwaterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,swmm_inp="data/simple1.inp"):


        super(StormwaterEnv, self).__init__()
        self.total_rewards = []
        self.eps_reward = 0

        self.num_steps = 0

        self.timestep = 0

        self.swmm_inp = swmm_inp        
        self.temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
        self.temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting
       
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
        self.swmm_inp="data/simple1.inp"
        
        self.total_rewards.append(self.eps_reward)
        #self.swmm_inp = 'simple%s.inp'%(random.randint(1,3))
        self.current_step = 0

        self.timestep += 1
        self.eps_reward = 0
        
        self.num_steps = 0
        
        self.done = False

        
        self.sim = Simulation(self.swmm_inp)

        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)

        self.sim.start()
        #print(self.sim.end_time, self.sim.start_time)
        self.sim_len = self.sim.end_time - self.sim.start_time
        self.num_steps = int(self.sim_len.total_seconds()/self.control_time_step)
       # print(self.num_steps)

        self.link_object['R1'].target_setting = random.randrange(1)
        self.link_object['R2'].target_setting = random.randrange(1)
        self.link_object['R3'].target_setting = random.randrange(1)
        self.link_object['R4'].target_setting = random.randrange(1)
        self.link_object['R5'].target_setting = random.randrange(1)
        self.link_object['R6'].target_setting = random.randrange(1)


        self.settings = [
            self.link_object['R1'].current_setting,self.link_object['R2'].current_setting,
                self.link_object['R3'].current_setting,self.link_object['R4'].current_setting,
                    self.link_object['R5'].current_setting,self.link_object['R6'].current_setting]
       
        self.depths = [
            self.node_object['St1'].depth, self.node_object['St2'].depth,
                self.node_object['St3'].depth,self.node_object['St4'].depth,
                    self.node_object['St5'].depth,self.node_object['St6'].depth,
                         self.node_object['J2'].depth, self.node_object['J3'].depth,
                             self.node_object['J4'].depth]
       
        self.flooding = [
            self.node_object['St1'].flooding, self.node_object['St2'].flooding,
                self.node_object['St3'].flooding,self.node_object['St4'].flooding,
                    self.node_object['St5'].flooding,self.node_object['St6'].flooding,
                         self.node_object['J2'].flooding, self.node_object['J3'].flooding,
                             self.node_object['J4'].flooding]
        
     
        self.reward = reward_function(self.depths, self.flooding)
        
        self.eps_reward += self.reward
        
        state = []
        
        for i in self.settings:
            state.append(i)
        for i in self.depths:
            state.append(i)
        for i in self.flooding:
            state.append(i)
            
        return state


    def step(self, action):
        
        self.current_step += 1 # Decide whether this should be before or after taking actions
        
        self.link_object['R1'].target_setting = action[0]
        self.link_object['R2'].target_setting = action[1]
        self.link_object['R3'].target_setting = action[2]
        self.link_object['R4'].target_setting = action[3]
        self.link_object['R5'].target_setting = action[4]
        self.link_object['R6'].target_setting = action[5]

        self.sim.__next__()
        
        
        self.settings = [
            self.link_object['R1'].current_setting,self.link_object['R2'].current_setting,
                self.link_object['R3'].current_setting,self.link_object['R4'].current_setting,
                    self.link_object['R5'].current_setting,self.link_object['R6'].current_setting]
       
        self.depths = [
            self.node_object['St1'].depth, self.node_object['St2'].depth,
                self.node_object['St3'].depth,self.node_object['St4'].depth,
                    self.node_object['St5'].depth,self.node_object['St6'].depth,
                         self.node_object['J2'].depth, self.node_object['J3'].depth,
                             self.node_object['J4'].depth]
       
        self.flooding = [
            self.node_object['St1'].flooding, self.node_object['St2'].flooding,
                self.node_object['St3'].flooding,self.node_object['St4'].flooding,
                    self.node_object['St5'].flooding,self.node_object['St6'].flooding,
                         self.node_object['J2'].flooding, self.node_object['J3'].flooding,
                             self.node_object['J4'].flooding]
        
        self.reward = reward_function(self.depths, self.flooding)

        self.eps_reward += self.reward

        state = []
        for i in self.settings:
            state.append(i)
        for i in self.depths:
            state.append(i)
        for i in self.flooding:
            state.append(i)


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
        #print(self.log_dumps)
        pack = (random.sample(self.log_dumps,5))
        pack.append((self.log_dumps[len(self.log_dumps)-1]))
        for plot, dump in enumerate(pack):

            settings, depths, floodings = [[], []], [[], []], [[], [], []]
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

            plt.subplot(2, 3, 1)
            plt.plot(settings[0])
            plt.ylim(0, 1)
            plt.title('R1')
            plt.ylabel("Valve Opening")
            plt.xlabel("time step")

            plt.subplot(2, 3, 2)
            plt.plot(settings[1])
            plt.ylim(0, 1)
            plt.title('R2')
            plt.ylabel("Valve Opening")
            plt.xlabel("time step")

            plt.subplot(2, 3, 3)
            plt.plot(settings[2])
            plt.ylim(0, 1)
            plt.title('R3')
            plt.ylabel("Valve Opening")
            plt.xlabel("time step")

            plt.subplot(2, 3, 4)
            plt.plot(settings[3])
            plt.ylim(0, 1)
            plt.title('R4')
            plt.ylabel("Valve Opening")
            plt.xlabel("time step")

            plt.subplot(2, 3, 5)
            plt.plot(settings[4])
            plt.ylim(0, 1)
            plt.title('R5')
            plt.ylabel("Valve Opening")
            plt.xlabel("time step")

            plt.subplot(2, 3, 6)
            plt.plot(settings[5])
            plt.ylim(0, 1)
            plt.title('R6')
            plt.ylabel("Valve Opening")
            plt.xlabel("time step")

            plt.tight_layout()
            if(plot == 5):
                plt.savefig(location + "TEST_" + str(timestep) + "_STATES", dpi=300)
            else:
                plt.savefig(location + str(timestep) + "_STATES", dpi=300)
           
            plt.clf()
            
            plt.subplot(2, 3, 1)
            plt.plot(depths[0])
            plt.ylim(0, 5)
            plt.title('St1 Depth')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 3, 2)
            plt.plot(depths[1])
            plt.ylim(0, 5)
            plt.title('St2 Depth')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 3, 3)
            plt.plot(depths[2])
            plt.ylim(0, 5)
            plt.title('St3 Depth')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 3, 4)
            plt.plot(depths[3])
            plt.ylim(0, 5)
            plt.title('St4 Depth')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 3, 5)
            plt.plot(depths[4])
            plt.ylim(0, 5)
            plt.title('St5 Depth')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 3, 6)
            plt.plot(depths[5])
            plt.ylim(0, 5)
            plt.title('St6 Depth')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.tight_layout()
            if(plot == 5):
                plt.savefig(location + "TEST_" + str(timestep) + "_DEPTHS", dpi=300)
            else:
                plt.savefig(location + str(timestep) + "_DEPTHS", dpi=300)
           
            plt.clf()


            plt.subplot(2, 2, 1)
            plt.plot(floodings[0])
            #plt.ylim(0, 5)
            plt.title('J2 flooding')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 2, 2)
            plt.plot(floodings[1])
            # plt.ylim(0, 5)
            plt.title('J3 flooding')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 2, 3)
            plt.plot(floodings[2])
            # plt.ylim(0, 5)
            plt.title('J4 Flooding')
            plt.ylabel("Feet")
            plt.xlabel("time step")

            plt.subplot(2, 2, 4)
            plt.bar([0, 1, 2], [sum(floodings[0]), sum(floodings[1]), sum(floodings[2]),
                sum(floodings[3]), sum(floodings[4]), sum(floodings[5]),
                    sum(floodings[6]), sum(floodings[7]), sum(floodings[8])], tick_label=["St1","St2","St3",
                                                                                            "St4","St5","St6",
                                                                                                "J2", "J3", "J4"])
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
            plt.ylim(-500,0)
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