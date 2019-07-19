import numpy as np
import random
from reward_functions import reward_function2 as reward_function
from pyswmm import Simulation, Nodes, Links
#import matplotlib as plt


class Env:
    def reset(self):
        
   #     self.total_reward.append(self.eps_reward)
        
        self.swmm_inp = "simple_2_ctl_smt.inp"

        self.temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
        self.temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting
        
        self.eps_reward = 0
        self.St1_depth = []
        self.St2_depth = []
        self.J3_depth = []
        self.St1_flooding = []
        self.St2_flooding = []
        self.J3_flooding = []
        self.R1_position = []
        self.R2_position = []
        
        self.done = False

        
        self.sim = Simulation(self.swmm_inp)

        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)

        self.sim.start()
       # print(self.sim.end_time, self.sim.start_time)
        self.sim_len = self.sim.end_time - self.sim.start_time
        self.num_steps = int(self.sim_len.total_seconds()/self.control_time_step)
        

        self.link_object['R1'].target_setting = random.randrange(1)
        self.link_object['R2'].target_setting = random.randrange(1)


        self.settings = [self.link_object['R1'].current_setting,self.link_object['R2'].current_setting]
        self.depths = [self.node_object['St1'].depth,self.node_object['St2'].depth]
        self.flooding = [self.node_object['St1'].flooding, self.node_object['St2'].flooding, self.node_object['J3'].flooding]

     
        self.St1_depth.append(self.depths[0])
        self.St2_depth.append(self.depths[1])
        self.J3_depth.append(self.node_object['J3'])
        
        self.St1_flooding.append(self.flooding[0])
        self.St2_flooding.append(self.flooding[1])
        self.J3_flooding.append(self.flooding[2])
        
        self.R1_position.append(self.settings[0])
        self.R2_position.append(self.settings[1])
        
        




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

        
        
        self.link_object['R1'].target_setting = action[0]
        self.link_object['R1'].target_setting = action[1]

        self.sim.__next__()
        
        
        self.settings = [self.link_object['R1'].current_setting,self.link_object['R2'].current_setting]
        self.depths = [self.node_object['St1'].depth,self.node_object['St2'].depth]
        self.flooding = [self.node_object['St1'].flooding, self.node_object['St2'].flooding, self.node_object['J3'].flooding]
        
        self.reward = reward_function(self.depths, self.flooding)
        state = []
        for i in self.settings:
            state.append(i)
        for i in self.depths:
            state.append(i)
        for i in self.flooding:
            state.append(i)
    
        return state, self.reward, self.done
    
    def close(self):
        self.sim.report()
        self.sim.close()
'''  
    def graph(self):
        plt.subplot(2, 2, 1)
        plt.plot(self.St1_depth)
        plt.ylim(0, 5)
        plt.title('St1_depth')
        plt.ylabel("ft")
        plt.xlabel("time step")

        plt.subplot(2, 2, 2)
        plt.plot(self.St2_depth)
        plt.ylim(0, 5)
        plt.title('St2_depth')
        plt.ylabel("ft")
        plt.xlabel("time step")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.J3_depth)
        plt.ylim(0, 2)
        plt.title('J3_depth')
        plt.ylabel("ft")
        plt.xlabel("time step")
        
# bar graph for total flooding
        plt.subplot(2, 2, 4)
        plt.bar([0, 1, 2], [sum(self.St1_flooding), sum(self.St2_flooding), sum(self.J3_flooding)], tick_label=["St1", "St2", "J3"])
        plt.ylim(0)
        plt.title('total_flooding')
        plt.ylabel("10^3 cubic feet")
        
        plt.tight_layout()
        plt.show()
        plt.savefig("flood.png", dpi=300)
        #plt.close()
        
        # plot rewards and actions
        plt.subplot(2, 1, 1)
        plt.plot(self.total_reward)
        plt.ylabel("average reward")
        plt.xlabel("episode")
        
        plt.subplot(2, 1, 2)
        plt.plot(self.R1_position)
        plt.plot(self.R2_position, linestyle='--')
        plt.ylim(0, 1)
        plt.ylabel("orifice position")
        plt.xlabel("time step")
        plt.tight_layout()
        plt.savefig("pos.png", dpi=300)
        #plt.show()
        
        plt.close()
    '''


'''        



    
        '''
   
