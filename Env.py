import numpy as np
import random
from reward_functions import reward_function2 as reward_function

class Env:
    def reset(self):

        self.swmm_inp = "simple_2_ctl_smt.inp"

        self.temp_height = np.zeros(2, dtype='int32')  # St1.depth, St2.depth
        self.temp_valve = np.zeros(2, dtype='int32')  # R1.current_setting, R2.current_setting

        self.action_space = 2  # number of structures to control

        self.St1_depth = []
        self.St2_depth = []
        self.J3_depth = []
        self.St1_flooding = []
        self.St2_flooding = []
        self.J3_flooding = []
        self.R1_position = []
        self.R2_position = []
        self.done = True

        
        self.sim = Simulation(self.swmm_inp)

        self.control_time_step = 900  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        self.node_object = Nodes(self.sim)
        self.link_object = Links(self.sim)

        self.sim.start()
        self.sim_len = self.sim.end_time - self.sim.start_time
        self.num_steps = int(self.sim_len.total_seconds()/self.control_time_step)

        self.link_object['R1'].target_setting = random.randrange(1)
        self.link_object['R2'].target_setting = random.randrange(1)


        self.settings = [self.link_object['R1'].current_setting,self.link_object['R2'].current_setting]
        self.depths = [self.node_object['St1'].depth,self.node_object['St2'].depth]
        self.flooding = [self.node_object['St1'].flooding, self.node_object['St2'].flooding, self.node_object['J3'].flooding]

        self.reward = reward_function(self.depths, self.flooding)

        return [self.settings[0], self.depths[0], self.flooding[0]], self.reward, self.done


    def step(self, action):

        self.link_object['R1'].target_setting = action[0]
        self.link_object['R1'].target_setting = action[1]

        self.sim.__next__()

        self.settings = [self.link_object['R1'].current_setting,self.link_object['R2'].current_setting]
        self.depths = [self.node_object['St1'].depth,self.node_object['St2'].depth]
        self.flooding = [self.node_object['St1'].flooding, self.node_object['St2'].flooding, self.node_object['J3'].flooding]

        self.reward = reward_function(self.depths, self.flooding)

        return [self.settings, self.depths, self.flooding], self.reward, self.done

    def close(self):
        sim.report()
        sim.close()
